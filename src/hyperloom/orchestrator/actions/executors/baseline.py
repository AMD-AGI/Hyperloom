# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

``BenchmarkRunExecutor`` is the benchmark round shared by every action that runs one;
``BaselineExecutor`` adds the baseline timeline event, the benchmark watchdog and the
InferenceX eval patching.
"""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_benchmark_config
import asyncio
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, NamedTuple

import yaml

from hyperloom.common.coerce import optional_positive_float
from hyperloom.common.env import is_truthy
from hyperloom.common.gain_math import incremental_gain_pct
from hyperloom.common.env_safety import redact_secret_values, scrub_benchmark_process_env
from hyperloom.common.git_safety import safe_directory_args
from hyperloom.common.jsonio import read_json
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import agentx_active, is_agentx_mode, with_output_per_gpu
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    ROUND_ACCURACY,
    ROUND_MEASURE,
    ROUND_SINGLE,
    ROUND_WARMUP,
    RUN_AFTER_MOE_RUNNER_FAILURE,
    RUN_INITIAL,
    BaselineInvocation,
    BaselineRequest,
    BaselineRound,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer import inferencex_checkout
from hyperloom.inference_optimizer.grid_server_args import remove_moe_runner_backend_arg
from hyperloom.inference_optimizer.grid_server_args import with_capture_disabled
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...loop.sub_agent_runner import RunnerContext
from ...measurement.integrate_performance import assess_integrate_performance
from ...measurement.runtime_findings import persist_runtime_findings, scan_server_log
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows, report_progress
from ...phases import machine_state as _phase_state
from ...state.failure_evidence import (
    KV_CACHE_OOM_MARKERS,
    MAGPIE_NONZERO_INVALID_MEASUREMENT,
    SUBPROCESS_NONZERO,
    nonzero_output_signature,
)
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from . import _server_lifecycle as _lifecycle
from ._aiter_jit import is_aiter_jit_registry_mismatch
from ._launch_evidence import RequestedLaunch, build_launch_evidence, persist_launch_evidence

# The grid module is the namespace the helpers both benching arms share ended up in: how a sentinel returncode reads
# back, how a round's cap is clamped to the budget, how the two session bounds are resolved, and the hygiene every
# launch needs.
from ._grid_runner import (
    SessionDirField,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
    stopped_by_the_run,
    sync_benchmark_timeout,
)
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_ERROR_CLASS,
    DETOKENIZER_STALL_RETURNCODE,
    RoundWatch,
    clear_server_ready_stamp,
    post_ready_runtime_sec,
    resolve_benchmark_timeouts,
    run_with_session_kill,
    server_log_death_excerpt,
    session_deadline_to_remaining_sec,
)
from ._accuracy_gate import materialized_run_eval_disabled, post_ready_sec_without_eval

from ._workload_envs import (
    VLLM_SOURCE_ROOT_ENVS,
    _client_tokenizer_mode,
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
    prepare_agentx_runtime,
)
from ._inferencex_patcher import (
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_lib_eval_start_patched,
    ensure_benchmark_serving_seed_patched,
    ensure_eval_probe_patched,
    ensure_eval_unbound_outputs_patched,
    eval_probe_targets_exist,
    failed_patch_anchors,
    failed_patch_anchors_in,
)
from ._git import _git_head_sha
from ._agentx_prefix_cache import refuse_cacheless_agentx_baseline
from ._magpie_patcher import client_tokenizer_hook_installed, ensure_eval_concurrency_compat
from .benchmark_config import apply_benchmark_round_seed
from ._patch_snapshot import (
    _create_patch_snapshot,
    _patch_touched_paths_from_text,
    _restore_patch_snapshot,
)
from .benchmark_result import (
    double_run_requested,
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    select_run_workspace,
    served_complete_protocol,
    snapshot_workspaces,
)
from .benchmark_backend import build_benchmark_command
from .bypass_engine import lifecycle_meta_file, lifecycle_pid_file
from .magpie_agentx import ServerArgvInvalidError


log = logging.getLogger(__name__)
_BENCHMARK_MODE_ATTR = "benchmark_mode"

#: The producer label the baseline event's fragments are written under.
_RECORDER_PRODUCER = "orchestrator"

# Markers identifying an InferenceX ``run_eval`` (lm-eval) failure as the root cause of a benchmark non-zero exit.
_EVAL_FAILURE_MARKERS = (
    "run_eval failed with exit code",
    "ERROR: run_eval failed",
    "Unknown parameter: --concurrent-requests",
)
# Markers showing the eval failed because the server was not reachable, rather than because the model scored badly or
# the framework lacks a capability. The eval harness reports both the same way -- a non-zero ``run_eval`` -- so without
# this distinction a torn-down server is indistinguishable from an accuracy gap, and gets routed to the enablement lane
# to hunt for a capability that was never missing.
_EVAL_SERVER_UNREACHABLE_MARKERS = (
    "ClientConnectorError",
    "Cannot connect to host",
    "Connect call failed",
)
# Bounded per-file read so log scanning never slurps a multi-GB server.log.
_LOG_SCAN_MAX_BYTES = 262_144
# The measured pass ran as the first traffic against a freshly restarted server, because the warmup that exists to
# drive it did not.
_MN_WARMUP_DID_NOT_WARM_WARNING = "baseline_mn_warmup_did_not_run"
# The round kept its warmup pass as the baseline because the budget could not pay for the measured pass after it.
MEASURE_ROUND_DROPPED_WARNING = "baseline_measure_round_dropped_low_budget"

# The cold-start guard's two round directories.
_WARMUP_ROUND_DIR = "warmup_round"
_MEASURE_ROUND_DIR = "measure_round"
_DOUBLE_RUN_ROUND_DIRS = (_WARMUP_ROUND_DIR, _MEASURE_ROUND_DIR)
#: The staged accuracy pass's slot, run on the server the measured pass leaves hot.
_ACCURACY_ROUND_DIR = "accuracy_round"

#: What a round reaped by the detokenizer-stall watchdog reports: the server came up, then went silent.
_DETOKENIZER_STALL_ERROR = (
    "server reported ready but emitted no log output (hung engine / detokenizer stall); "
    "reaped by the detokenizer-stall watchdog. See server.log."
)

#: The file Magpie writes its own measurement to, inside the workspace it creates.
_BENCHMARK_REPORT_NAME = "benchmark_report.json"

# Markers identifying a MoE quant scheme with no implementation for the ``--moe-runner-backend`` in use:
# ``create_moe_runner`` falls through without building a runner and the first forward pass dies (e.g. Quark MXFP4 on
# triton).
_MOE_RUNNER_MISSING_MARKERS = (
    "has no attribute 'runner'",
    'has no attribute "runner"',
)
_MOE_SCHEME_CONTEXT_MARKERS = (
    "create_moe_runner",
    "moe_runner",
    "_moe.py",
    "fused_moe",
    "/moe/",
)


# Time limit for heuristic CLI-error matching; explicit configuration validation is independent of runtime.
FAST_EXIT_THRESHOLD_SEC = 30.0
_ARG_ERROR_CLASS = "fast_exit_arg_error"
_ARG_ERROR_PATTERNS = (
    "unrecognized arguments",
    "invalid choice",
    "Unknown attention backend",
    "not a valid",
)
_ARG_ERROR_CONTEXT_PATTERNS = (
    "argument",
    "argparse",
    "backend",
    "choice",
    "cli",
    "invalid",
    "option",
    "flag",
    "unknown",
)


# Strong cuda-graph capture markers: stream-capture incompatibility, reliably recoverable by disabling cuda-graph.
_CUDA_GRAPH_STRONG_MARKERS = (
    "operation not permitted when stream is capturing",
    "hiperrorstreamcaptureunsupported",
)
# Weak marker: bare "Capture cuda graph failed" carries no root cause.
_CUDA_GRAPH_WEAK_MARKER = "capture cuda graph failed"

# Profile-cuda-graph shape discovery triggers a recoverable AssertionError that must win over the generic
# assertionerror non-recoverable gate below.
_CUDA_GRAPH_PROFILE_ASSERT_MARKERS = (
    "get_num_new_pages",
    "seq_lens.device == cpu_device",
)

# OOM-rooted capture failures are NOT recoverable by disabling cuda-graph (eager peaks can be higher);
# compile/lowering errors are not either.
_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
)
_NON_RECOVERABLE_MARKERS = (
    "loweringexception",
    "assertionerror",
    "compilationerror",
)
# Strong markers are high-confidence, so OOM exclusion is scoped tight (±1 line): only an OOM on/adjacent to the
# marker line demotes it.
_STRONG_OOM_CONTEXT_RADIUS = 1


#: Subdirectory of a round slot holding earlier attempts' server logs.
_ATTEMPTS_DIRNAME = "attempts"

#: File in that subdirectory holding the slot's next attempt index. An attempt
#: that produced no log retains nothing, so the index cannot be counted off the
#: retained directories.
_ATTEMPT_COUNTER_NAME = "next_index"

# Both timestamp shapes a served model writes: SGLang stamps a full date,
# vLLM's default formatter omits the year.
_SERVER_LOG_CLOCK = re.compile(r"(?:(\d{4})-)?(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")

#: Substituted for a year the log did not print. A leap year, so a Feb-29 line
#: still parses; only the difference between two stamps is used.
_CLOCK_ASSUMED_YEAR = 2000


@dataclass(frozen=True)
class BringupLog:
    """One bring-up log as far as it could be read.

    Attributes:
        text: The decoded log; empty when no log was written and empty when
            the log could not be read.
        degraded: Empty when the read answered for the log's contents;
            :data:`~hyperloom.orchestrator.bringup.DEGRADED_UNREADABLE` when a
            log that exists could not be read back.
    """

    text: str
    degraded: str = ""


def read_bringup_log(path: Path) -> BringupLog:
    """Read a server log whole for bring-up classification.

    Neither what the ladder looks for nor where it lies is bounded: a build that
    logs per kernel shape pushes the later milestones megabytes in, and a worker's
    root traceback can sit megabytes above the engine-dead consequence that ends
    the log. The read is linear in the log's size, so callers run it off the
    event loop.

    A log that was never written and a log the mount refuses to serve are
    different answers, and both are answers: an ESTALE or EIO on the session
    mount degrades the observation rather than costing the round.

    Args:
        path: The log file to read.

    Returns:
        BringupLog: The decoded text and the degraded outcome when a log that
        exists could not be read.
    """
    from ...bringup import DEGRADED_UNREADABLE

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return BringupLog("")
    except OSError as exc:
        log.warning("bringup: server log %s could not be read (%s)", path, exc)
        return BringupLog("", DEGRADED_UNREADABLE)
    return BringupLog(data.decode("utf-8", "replace"))


def server_child_elapsed_sec(server_log_text: str) -> float:
    """Return how long the server child ran, on the server child's own clock.

    Taken from the log's timestamps rather than the wrapper's spawn-to-return
    wall-clock, which also covers materialisation, the client and teardown.

    Args:
        server_log_text: Text read from the server child's log.

    Returns:
        float: Seconds between the first and last timestamped line, or ``0.0``
        when fewer than two lines carry a parseable timestamp.
    """
    stamps: list[datetime] = []
    for match in _SERVER_LOG_CLOCK.finditer(server_log_text):
        year, month, day, hour, minute, second = match.groups()
        try:
            stamps.append(
                datetime(
                    int(year or _CLOCK_ASSUMED_YEAR),
                    int(month),
                    int(day),
                    int(hour),
                    int(minute),
                    int(second),
                )
            )
        except ValueError:
            continue
    if len(stamps) < 2:
        return 0.0
    return max(0.0, (stamps[-1] - stamps[0]).total_seconds())


def open_bringup_attempt(output_dir: Path) -> int:
    """Retain the previous attempt's ``server.log`` and index this attempt.

    A round slot is reused across retries, so the previous attempt's log is
    moved into its own attempt directory, gzipped, rather than deleted.

    Args:
        output_dir: The per-round workspace slot.

    Returns:
        int: This attempt's zero-based index within the slot.

    Raises:
        OSError: When the previous log cannot be rotated out; a log left in
            place would be classified as this attempt's.
    """
    attempts = output_dir / _ATTEMPTS_DIRNAME
    index = _claim_attempt_index(attempts)
    live = output_dir / "server.log"
    if not live.exists():
        return index
    # The live log is the *previous* attempt's, hence the index below.
    kept = attempts / f"{max(index - 1, 0):03d}"
    kept.mkdir(parents=True, exist_ok=True)
    with live.open("rb") as source, gzip.open(kept / "server.log.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    live.unlink()
    return index


def _claim_attempt_index(attempts: Path) -> int:
    """Claim the next attempt index for a round slot and persist the successor.

    Args:
        attempts: The slot's attempts directory.

    Returns:
        int: The claimed zero-based index; ``0`` for the slot's first attempt.

    Raises:
        OSError: When the counter cannot be read or advanced; two attempts
            would otherwise claim one index and overwrite one observation.
        ValueError: When the counter file holds something that is not an index.
    """
    counter = attempts / _ATTEMPT_COUNTER_NAME
    try:
        index = max(0, int(counter.read_text(encoding="utf-8").strip() or "0"))
    except FileNotFoundError:
        index = 0
    attempts.mkdir(parents=True, exist_ok=True)
    counter.write_text(f"{index + 1}\n", encoding="utf-8")
    return index


#: Capture-failure categories. ``instrumentation`` is the profiler's own shape discovery colliding with capture --
#: the same server args capture cleanly with no profiler attached, so the fix belongs to the instrumentation.
#: ``config`` is the server args themselves failing to capture. Roofline reports the category instead of acting on
#: it; only baseline still uses the boolean, to decide its one-shot retry.
CUDA_GRAPH_CAPTURE_INSTRUMENTATION = "instrumentation"
CUDA_GRAPH_CAPTURE_CONFIG = "config"


def _classify_cuda_graph_capture_failure(*texts: str) -> tuple[str, str]:
    """Classify a capture failure as ``(category, matched_marker)``; empty category means no marker matched.

    The marker is reported alongside the category because the match is a string heuristic: an offline reader has to
    be able to see what fired and overrule it.
    """
    lines = "\n".join(t for t in texts if t).splitlines()
    lowered = [ln.lower() for ln in lines]
    blob = "\n".join(lowered)
    # Profile-cuda-graph assert wins over the assertionerror gate.
    if all(m in blob for m in _CUDA_GRAPH_PROFILE_ASSERT_MARKERS):
        return CUDA_GRAPH_CAPTURE_INSTRUMENTATION, " + ".join(_CUDA_GRAPH_PROFILE_ASSERT_MARKERS)
    blob_has_oom = any(m in blob for m in _OOM_MARKERS)
    blob_has_non_recoverable = any(m in blob for m in _NON_RECOVERABLE_MARKERS)
    saw_pure_weak = False
    for idx, line in enumerate(lowered):
        strong = next((m for m in _CUDA_GRAPH_STRONG_MARKERS if m in line), "")
        if strong:
            lo = max(0, idx - _STRONG_OOM_CONTEXT_RADIUS)
            hi = min(len(lowered), idx + _STRONG_OOM_CONTEXT_RADIUS + 1)
            if not any(m in "\n".join(lowered[lo:hi]) for m in _OOM_MARKERS):
                return CUDA_GRAPH_CAPTURE_CONFIG, strong
            continue
        if _CUDA_GRAPH_WEAK_MARKER in line:
            saw_pure_weak = True
    if saw_pure_weak and not blob_has_oom and not blob_has_non_recoverable:
        return CUDA_GRAPH_CAPTURE_CONFIG, _CUDA_GRAPH_WEAK_MARKER
    return "", ""


def _is_cuda_graph_capture_failure(*texts: str) -> bool:
    """True when a cuda-graph capture marker is recoverable by disabling graph."""
    return bool(_classify_cuda_graph_capture_failure(*texts)[0])


# Disable cuda-graph capture per framework: sglang uses --disable-cuda-graph, vllm uses the dotted
# --compilation-config.cudagraph_mode NONE (which keeps inductor), atom still uses --enforce-eager.
_DISABLE_CUDA_GRAPH_FLAGS = {
    "sglang": "--disable-cuda-graph",
    "vllm": "--compilation-config.cudagraph_mode NONE",
    "atom": "--enforce-eager",
}


async def _prepare_aiter_serving_so(extra_envs: dict[str, Any]) -> None:
    """Rebuild the serving ``.so`` before boot when the CSVs it loads name kernels it lacks.

    sglang starts against whatever ``get_config_file`` resolves each ``AITER_CONFIG_*`` to,
    and a kernel the compiled module never registered raises from inside graph capture. The
    classifier downstream names that ``aiter_jit_registry_mismatch`` but cannot undo it, so
    the round is simply lost -- and PRELUDE's first measurement and every FRAMEWORK variant
    boot through here. The GEMM integrate lane already runs this check before its own boots.

    It is not conditioned on the round carrying a tuned CSV: a variable the round leaves
    unset is exactly the case where aiter merges the model overlays, which is where the
    kernel that fails the boot comes from.

    Nothing restores what this drops, so it is parked on the cache's own filesystem (the
    preflight's default) rather than copied into the run directory, which sits on shared NFS.

    Args:
        extra_envs: The round's environment, whose ``AITER_CONFIG_*`` values decide which
            branch of aiter's resolution each table takes.
    """
    csv_envs = {
        str(key): str(value)
        for key, value in (extra_envs or {}).items()
        if str(key).startswith("AITER_CONFIG_") and str(value).strip()
    }
    from ._aiter_jit import prepare_serving_so_for_csvs

    try:
        outcome = await asyncio.to_thread(prepare_serving_so_for_csvs, csv_envs)
    except OSError as exc:
        # A jit directory this cannot read is not a reason to lose the measurement.
        log.warning("baseline_executor: aiter serving .so preflight failed: %s", exc)
        return
    if isinstance(outcome, dict) and outcome.get("action") == "invalidate":
        log.info(
            "baseline_executor: the CSVs this boot loads outran the serving .so; dropped %d module(s) for rebuild",
            len(outcome.get("removed") or []),
        )


def _config_framework(config_path: Path | str) -> str:
    """Framework name from a materialized benchmark YAML (``""`` when unreadable)."""
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return ""
    return str((cfg.get("benchmark") or {}).get("framework") or "").strip().lower()


def _attach_baseline_launch_evidence(
    result: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    framework: str,
) -> None:
    """Persist the declared and observed server identity for a baseline result."""
    from ._grid_runner import _measurement_server_log_path

    workspace = Path(str(result.get("workspace") or "")) if result.get("workspace") else None
    actual_log = _measurement_server_log_path(output_dir / "server.log", workspace, slot=output_dir)
    result["server_log_path"] = actual_log or ""
    if result.get("benchmark_backend") == "magpie_agentx":
        from hyperloom.inference_optimizer.agentx.report import native_launch_evidence

        evidence = native_launch_evidence(result, config_path, actual_server_log=actual_log)
        result["launch_evidence"] = evidence
        result["launch_evidence_path"] = persist_launch_evidence(evidence, slot=output_dir)
        return
    evidence = build_launch_evidence(
        RequestedLaunch(config_path, framework), actual_server_log=actual_log, slot=output_dir
    )
    result["launch_evidence"] = evidence
    result["launch_evidence_path"] = persist_launch_evidence(evidence, slot=output_dir)
    declared_env = evidence["requested_server_env"].keys() - set(VLLM_SOURCE_ROOT_ENVS)
    report = scan_server_log(actual_log, evidence["framework"], declared_env=declared_env)
    persist_runtime_findings(report, slot=output_dir)


def _watchdog_server_log_path(output_dir: Path, framework: str) -> str | None:
    """``server.log`` path for the subprocess watchdogs, or None when server-less."""
    from hyperloom.inference_optimizer import framework_registry

    if framework_registry.is_scriptable(framework):
        return None
    return str(output_dir / "server.log")


def _round_post_ready_sec(
    server_log_path: str | None,
    *,
    started_unix: float,
    runtime_sec: float,
) -> float | None:
    """How much of a round's wall-clock was the benchmark rather than the boot."""
    if not server_log_path:
        return None
    return post_ready_runtime_sec(
        server_log_path,
        started_unix=started_unix,
        runtime_sec=runtime_sec,
    )


class _RoundEnd(NamedTuple):
    """How far a round the run stopped got: its returncode, the stop detail and its wall-clock."""

    returncode: int | None
    error: str
    runtime_sec: float


def _round_stop_result(
    stopped: StoppedByTheRun,
    end: _RoundEnd,
    *,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the failed result of a round the run itself stopped."""
    return {
        "status": "failed",
        "error_class": stopped.error_class,
        "returncode": end.returncode,
        "error": end.error,
        "subprocess_runtime_sec": round(end.runtime_sec, 2),
        "output_dir": str(output_dir),
        **capture_meta,
    }


def _stopped_round_result(
    returncode: int | None,
    *,
    round_label: str,
    runtime_sec: float,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the result for a round the run itself stopped, or ``None`` when it did not."""
    stopped = stopped_by_the_run(returncode)
    if stopped is None:
        return None
    log.warning(
        "baseline_executor: %s reaped after %.1fs: %s; error_class=%s.",
        round_label,
        runtime_sec,
        stopped.interrupted,
        stopped.error_class,
    )
    return _round_stop_result(
        stopped,
        _RoundEnd(returncode=returncode, error=stopped.interrupted, runtime_sec=runtime_sec),
        output_dir=output_dir,
        capture_meta=capture_meta,
    )


def _unlaunched_round_result(
    stopped: StoppedByTheRun,
    *,
    round_label: str,
    output_dir: Path,
    capture_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the result for a round the run stopped before launching it."""
    log.warning(
        "baseline_executor: %s not launched: %s; error_class=%s.",
        round_label,
        stopped.never_started,
        stopped.error_class,
    )
    return _round_stop_result(
        stopped,
        _RoundEnd(returncode=None, error=stopped.never_started, runtime_sec=0.0),
        output_dir=output_dir,
        capture_meta=capture_meta,
    )


def _round_headroom_sec(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds this round's budget may still spend, and the numbers behind it."""
    if state is None:
        return None, {"reason": "no_session_state"}
    usable_sec = _phase_state.session_usable_seconds(state)
    if usable_sec is None:
        return None, {"reason": "unbounded_session_budget"}
    return usable_sec, {"bound": "session_usable", "affordable_sec": round(usable_sec, 1)}


def _cold_anchor_from_warmup(
    warmup_result: dict[str, Any],
    *,
    dropped: dict[str, Any],
) -> dict[str, Any]:
    """Keep the warmup's cold figure as the anchor, marked as the cold one."""
    warnings = warmup_result.setdefault("nonfatal_warnings", [])
    if MEASURE_ROUND_DROPPED_WARNING not in warnings:
        warnings.append(MEASURE_ROUND_DROPPED_WARNING)
    warmup_result["measure_round_dropped"] = dropped
    return warmup_result


def _a_use_must_follow_the_round(state: Any) -> bool:
    """Whether this round is only worth running if something can be measured after it."""
    phase = str(getattr(state, "phase", "") or "").strip().upper()
    return phase == _phase_state.PHASE_PRELUDE


def _disable_cuda_graph_flag(framework: str) -> str:
    """Return the framework-correct flag that disables cuda-graph capture."""
    return _DISABLE_CUDA_GRAPH_FLAGS.get(
        (framework or "").strip().lower(),
        "--disable-cuda-graph",
    )


def _with_cuda_graph_disabled(extra_server_args: str, framework: str) -> str:
    """Apply the framework-correct disable-cuda-graph flag once (idempotent)."""
    return with_capture_disabled(extra_server_args or "", _disable_cuda_graph_flag(framework))


def _classify_subprocess_error(elapsed_sec: float, stderr_tail: str) -> str:
    """Recognize KV-cache OOM, configuration validation, and fast argument failures."""
    tail = (stderr_tail or "").lower()
    # KV-cache OOM can surface long after weight load; match before the fast-exit elapsed gate below.
    if any(m in tail for m in KV_CACHE_OOM_MARKERS):
        return "kv_cache_oom"
    # Config validation can follow slow framework imports; elapsed time does not change its cause.
    if re.search(r"\bvalidation errors? for cacheconfig\b", tail):
        return _ARG_ERROR_CLASS
    if elapsed_sec >= FAST_EXIT_THRESHOLD_SEC:
        return SUBPROCESS_NONZERO
    if any(p.lower() in tail for p in _ARG_ERROR_PATTERNS):
        return _ARG_ERROR_CLASS
    if "valueerror:" in tail and any(p in tail for p in _ARG_ERROR_CONTEXT_PATTERNS):
        return _ARG_ERROR_CLASS
    return SUBPROCESS_NONZERO


def _set_materialized_run_eval(config_path: Path, *, enabled: bool) -> None:
    """Set the effective eval mode after lifecycle eligibility is known."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark = cfg.setdefault("benchmark", {})
    benchmark.setdefault("envs", {})["RUN_EVAL"] = "true" if enabled else "false"
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_result_dir(output_dir: Path, override_result_dir: str | None) -> Path:
    """Resolve the benchmark ``$RESULT_DIR`` exactly as the subprocess sees it."""
    if not override_result_dir:
        return output_dir
    result_dir = Path(override_result_dir)
    if result_dir.is_absolute():
        return result_dir
    return (output_dir / result_dir).resolve()


def _benchmark_process_env(
    config_path: Path, output_dir: Path, result_dir: Path, inferencex_path: str
) -> dict[str, str]:
    """Build the client environment without overriding Magpie's native checkout selection."""
    from .magpie_agentx import native_config

    env = scrub_benchmark_process_env(os.environ.copy())
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    if inferencex_path and not native_config(config_path):
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
    env["RESULT_DIR"] = str(result_dir)
    # InferenceX removes EVAL_RESULT_DIR after processing lm-eval output.
    env["EVAL_RESULT_DIR"] = str(result_dir / "eval_output")
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")
    return env


def _should_establish_quality_ref(task_kind: str | None, params: dict[str, Any] | None = None) -> bool:
    """Only a genuine ``baseline`` task may establish/overwrite the quality reference."""
    if str(task_kind or "") != "baseline":
        return False
    return not (params or {}).get("quality_ref_exempt")


# Above this cold-start delta the measured round is unlikely to have settled either: the observed pathological case
# climbed 14,202 -> 19,374 -> 22,425 tok/s across three rounds of one unchanged config, i.e. +36% into round 2 and
# another +16% into round 3.
_COLD_START_DELTA_WARN_PCT = 25.0


def _requested_framework(params: dict[str, Any]) -> str | None:
    return str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None


def _is_double_run_accuracy_handoff(
    result: dict[str, Any],
    salvaged: dict[str, Any] | None,
) -> bool:
    """Whether accuracy came from the warmup round because that is the design."""
    out_dir = str((result or {}).get("output_dir") or "")
    if not out_dir or Path(out_dir).name != _MEASURE_ROUND_DIR:
        return False
    source = str((salvaged or {}).get("source_file") or "")
    return _WARMUP_ROUND_DIR in Path(source).parts


def _git_toplevel(repo_path: str) -> str:
    """Return the work-tree root of ``repo_path``, or ``\"\"`` when it is not in one."""
    if not repo_path:
        return ""
    try:
        result = subprocess.run(
            ["git", *safe_directory_args(["rev-parse", "--show-toplevel"], cwd=repo_path)],
            cwd=repo_path,
            capture_output=True,
            timeout=5,
            check=True,
        )
        return result.stdout.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def _patch_present_in_committed_head(
    repo_path: str,
    patch_path: Path,
) -> bool:
    """Check reverse applicability against a temporary index loaded from HEAD."""
    fd, raw_index = tempfile.mkstemp(prefix="warm-head-index-")
    os.close(fd)
    index_path = Path(raw_index)
    index_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = str(index_path)
    try:
        subprocess.run(
            ["git", "read-tree", "HEAD"],
            cwd=repo_path,
            env=env,
            capture_output=True,
            timeout=15,
            check=True,
        )
        reverse = subprocess.run(
            [
                "git",
                "apply",
                "-R",
                "--check",
                "--cached",
                str(patch_path),
            ],
            cwd=repo_path,
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return reverse.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    finally:
        index_path.unlink(missing_ok=True)
        Path(f"{index_path}.lock").unlink(missing_ok=True)


def _revert_patches(
    repo_path: str,
    pre_sha: str = "",
    snapshot_manifest: Any = None,
) -> dict[str, Any]:
    """Restore exact patch-touched state without broad reset/clean."""
    manifest, error = _validated_restore_manifest(repo_path, pre_sha, snapshot_manifest)
    result = {"ok": False, "errors": [error]} if error else _restore_patch_snapshot(manifest)
    if not result["ok"]:
        log.warning("baseline_executor: exact patch restore failed: %s", result["errors"])
    return result


def _validated_restore_manifest(
    repo_path: str,
    pre_sha: str,
    snapshot_manifest: Any,
) -> tuple[dict[str, Any], str]:
    """Load the snapshot manifest and check it was taken of ``repo_path`` at ``pre_sha``.

    Returns ``(manifest, "")`` when the restore may proceed, else ``({}, error)``.
    """
    manifest = snapshot_manifest
    if isinstance(manifest, (str, Path)):
        try:
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            return {}, f"manifest_read:{exc}"
    if not isinstance(manifest, dict):
        return {}, "missing_manifest"
    manifest_repo_value = str(manifest.get("repo_path") or "").strip()
    if not manifest_repo_value:
        return {}, "missing_manifest_repo"
    try:
        caller_repo = Path(repo_path).resolve(strict=True)
        manifest_repo = Path(manifest_repo_value).resolve(strict=True)
    except (OSError, ValueError) as exc:
        return {}, f"repo_validation:{type(exc).__name__}:{exc}"
    if caller_repo != manifest_repo:
        return {}, f"repo_mismatch:caller={caller_repo}:manifest={manifest_repo}"
    if pre_sha:
        try:
            head = subprocess.run(
                [
                    "git",
                    *safe_directory_args(
                        ["rev-parse", "HEAD"],
                        cwd=caller_repo,
                    ),
                ],
                cwd=caller_repo,
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            return {}, f"head_validation:{type(exc).__name__}:{exc}"
        if head != pre_sha:
            return {}, f"head_mismatch:expected={pre_sha}:actual={head}"
    return manifest, ""


def _three_way_residue_snapshot(
    repo_path: str,
    touched: list[str],
) -> dict[str, Any]:
    """Capture pre-existing residue only for this patch's paths."""
    root = Path(repo_path).resolve()
    unmerged = (
        subprocess.run(
            [
                "git",
                *safe_directory_args(
                    ["ls-files", "-u", "--", *touched],
                    cwd=repo_path,
                ),
            ],
            cwd=repo_path,
            capture_output=True,
            timeout=15,
            check=True,
        )
        .stdout.decode(errors="replace")
        .splitlines()
    )
    markers = (b"<<<<<<< ", b"=======", b">>>>>>> ")
    rows: dict[str, Any] = {}
    for rel in touched:
        target = root / rel
        marker_lines: list[str] = []
        if target.is_file() and not target.is_symlink():
            marker_lines = [
                line.decode(errors="replace") for line in target.read_bytes().splitlines() if line.startswith(markers)
            ]
        rows[rel] = {
            "reject": (root / f"{rel}.rej").exists(),
            "markers": marker_lines,
        }
    return {"unmerged": unmerged, "paths": rows}


def _verify_three_way_clean(
    repo_path: str,
    touched: list[str],
    before: dict[str, Any],
) -> tuple[bool, str]:
    """Reject only residue newly introduced on this patch's paths."""
    try:
        after = _three_way_residue_snapshot(repo_path, touched)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return False, f"post_3way_check_failed:{type(exc).__name__}"
    if set(after["unmerged"]) - set(before.get("unmerged") or []):
        return False, "new_unmerged_index_entries"
    for rel in touched:
        prior = (before.get("paths") or {}).get(rel) or {}
        now = after["paths"][rel]
        if now["reject"] and not prior.get("reject"):
            return False, f"new_reject_file:{rel}.rej"
        if set(now["markers"]) - set(prior.get("markers") or []):
            return False, f"new_conflict_marker:{rel}"
    return True, ""


def _revert_warm_patch_state(
    target_repo: str,
    *,
    pre_sha: str = "",
    snapshot_manifest: Any = None,
    nogit_backups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Restore warm-replay patch mutations via git snapshot or nogit backups."""
    if nogit_backups:
        from ...delivery.ledger import restore_records

        _restored, errors = restore_records(nogit_backups)
        if errors:
            log.warning("baseline_executor: nogit patch restore failed: %s", errors)
        return {"ok": not errors, "errors": errors}
    return _revert_patches(target_repo, pre_sha, snapshot_manifest)


#: The per-tree fields that leave this module. ``use_nogit`` stays behind: it
#: describes how the apply ran, and the restore reads the channel off whether
#: backups are present. ``mutated`` says whether this round wrote to the tree at
#: all, which absent restore artifacts alone cannot: a no-op apply and an apply
#: whose artifacts were lost both record none, and only the first is safe to
#: leave standing.
_WARM_TREE_FIELDS = ("root", "pre_sha", "snapshot_manifest", "nogit_backups", "mutated")


def _warm_tree_records(
    trees: Mapping[str, Mapping[str, Any]],
    *,
    before_mutation: bool = False,
) -> list[dict[str, Any]]:
    """Return one JSON-safe record per touched tree, in apply order.

    ``before_mutation`` stamps ``mutated`` true. A record persisted ahead of the apply is
    the one record that cannot know what the round went on to write, so a resume that finds
    it has to treat the tree as written-to and restore it from the snapshot taken with it.
    """
    records = [{field: tree[field] for field in _WARM_TREE_FIELDS} for tree in trees.values()]
    if before_mutation:
        for record in records:
            record["mutated"] = True
    return records


def _revert_warm_patch_trees(trees: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Restore every tree that was snapshotted, reporting the combined outcome."""
    errors: list[str] = []
    restored: list[str] = []
    for tree in trees:
        root = str(tree.get("root") or "")
        backups = list(tree.get("nogit_backups") or [])
        manifest = tree.get("snapshot_manifest")
        if not backups and not manifest:
            continue
        outcome = _revert_warm_patch_state(
            root,
            pre_sha=str(tree.get("pre_sha") or ""),
            snapshot_manifest=manifest,
            nogit_backups=backups,
        )
        if outcome.get("ok"):
            restored.append(root)
            continue
        errors.extend(f"{root}:{error}" for error in (outcome.get("errors") or ["restore_failed"]))
    return {"ok": not errors, "errors": errors, "restored": restored}


class _OverlayText(NamedTuple):
    content: str
    reason: str


def _refused_warm_replay(patches: list[Any], failure: str) -> dict[str, Any]:
    return {
        "required": True,
        "status": "failed",
        "patches": [],
        "applied": [],
        "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
        "failure": failure,
        "rolled_back": True,
    }


def _warm_patch_trees(roots: list[str]) -> dict[str, dict[str, Any]]:
    from ._nogit_patch import _is_git_tree

    # Two recorded roots inside one checkout are one tree: a git diff's paths are work-tree-root relative, so both
    # resolve there and applying them separately would ignore whatever falls outside the narrower one.
    for idx, root in enumerate(roots):
        if root and (toplevel := _git_toplevel(root)) and Path(toplevel).is_dir():
            roots[idx] = toplevel
    trees: dict[str, dict[str, Any]] = {}
    # Ordered by first use: the single-tree result fields report the first recorded root's tree, so that one has to
    # stay primary.
    for root in dict.fromkeys(root for root in roots if root):
        if not Path(root).is_dir():
            continue
        git_tree = _is_git_tree(Path(root))
        pre_sha = _git_head_sha(root) if git_tree else ""
        trees[root] = {
            "root": root,
            "pre_sha": pre_sha,
            "use_nogit": not git_tree or not pre_sha,
            "snapshot_manifest": None,
            "nogit_backups": [],
            "mutated": False,
        }
    return trees


def _warm_patch_text(patch: Mapping[str, Any]) -> _OverlayText:
    content = str(patch.get("patch_content") or "")
    patch_ref = str(patch.get("patch_ref") or "")
    patch_file = str(patch.get("patch_file") or "")
    if content:
        return _OverlayText(content, "")
    if not patch_ref:
        log.warning("baseline_executor: patch entry has no content/ref, skipping: %s", patch_file)
        return _OverlayText("", "missing_artifact")
    try:
        return _OverlayText(Path(patch_ref).read_text(encoding="utf-8", errors="replace"), "")
    except FileNotFoundError:
        log.warning("baseline_executor: patch_ref not found: %s", patch_ref)
        return _OverlayText("", "missing_artifact")
    except OSError as exc:
        log.warning("baseline_executor: cannot read patch_ref %s: %s", patch_ref, exc)
        return _OverlayText("", "artifact_read_failed")


def _unusable_patch_reason(content: str, patch_file: str) -> str:
    from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree

    if not is_unified_diff(content) or "GIT binary patch" in content:
        log.warning("baseline_executor: skipping warm patch %s — not a unified diff", patch_file)
        return "unsafe_or_non_text_diff"
    escape = patch_escapes_tree(content)
    if escape is not None:
        log.warning("baseline_executor: skipping warm patch %s — path escapes tree: %r", patch_file, escape)
        return "path_escapes_tree"
    if not _patch_touched_paths_from_text(content):
        return "missing_touched_paths"
    return ""


def _screened_overlays(patches: list[Any]) -> list[_OverlayText]:
    overlays: list[_OverlayText] = []
    for patch in patches:
        read = _warm_patch_text(patch)
        reason = read.reason or _unusable_patch_reason(read.content, str(patch.get("patch_file") or ""))
        overlays.append(_OverlayText(read.content, reason))
    return overlays


def _refuse_warm_replay_before_apply(
    patches: list[Any],
    roots: list[str],
    trees: dict[str, dict[str, Any]],
    overlays: list[_OverlayText],
) -> dict[str, Any] | None:
    if absent := next((root for root in roots if root and root not in trees), ""):
        return {**_refused_warm_replay(patches, "apply_root_absent"), "pre_sha": "", "target_repo": absent}
    rejected = next((idx for idx, overlay in enumerate(overlays) if overlay.reason), -1)
    if rejected < 0:
        return None
    primary = trees[next(iter(trees))]
    ref, reason = str(patches[rejected].get("patch_file") or ""), overlays[rejected].reason
    return {
        **_refused_warm_replay(patches, reason),
        "patches": [{"patch_ref": ref, "status": "failed", "reason": reason}],
        "failed_ref": ref,
        "pre_sha": primary["pre_sha"],
        "target_repo": primary["root"],
        "rolled_back": False,
    }


def _snapshot_warm_patch_trees(
    trees: dict[str, dict[str, Any]],
    roots: list[str],
    overlays: list[_OverlayText],
    output_dir: Path,
) -> None:
    by_tree: dict[str, list[str]] = {root: [] for root in trees}
    for idx, overlay in enumerate(overlays):
        if not overlay.reason and roots[idx] in trees:
            by_tree[roots[idx]].append(overlay.content)
    for position, root in enumerate(trees):
        tree = trees[root]
        if tree["use_nogit"] or not by_tree[root]:
            continue
        # _create_patch_snapshot owns one fixed sub-directory per output dir and clears it, so trees past the
        # first are given their own.
        tree["snapshot_manifest"] = _create_patch_snapshot(
            root,
            by_tree[root],
            output_dir if position == 0 else output_dir / "warm_patch_trees" / f"{position:02d}",
        )


def _git_apply_overlay(
    target_repo: str,
    patch_path: Path,
    patch_content: str,
    *,
    required_timeline: bool,
) -> str:
    checked = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=target_repo,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if checked.returncode == 0:
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=target_repo,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return "applied"
    if not required_timeline:
        raise RuntimeError(
            checked.stderr.decode(errors="replace")[:500] if checked.stderr else "git apply --check failed"
        )
    reverse = subprocess.run(
        ["git", "apply", "-R", "--check", str(patch_path)],
        cwd=target_repo,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if reverse.returncode == 0:
        return (
            "already_present"
            if _patch_present_in_committed_head(target_repo, patch_path)
            else "present_in_dirty_worktree"
        )
    touched = _patch_touched_paths_from_text(patch_content)
    before_residue = _three_way_residue_snapshot(target_repo, touched)
    three_way = subprocess.run(
        ["git", "apply", "--3way", str(patch_path)],
        cwd=target_repo,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if three_way.returncode != 0:
        raise RuntimeError(
            three_way.stderr.decode(errors="replace")[:500] if three_way.stderr else "git apply --3way failed"
        )
    clean, residue = _verify_three_way_clean(target_repo, touched, before_residue)
    if not clean:
        raise RuntimeError(residue)
    return "applied_3way"


def _apply_warm_patch_sequence(
    patches: list[Any],
    patch_trees: list[dict[str, Any] | None],
    overlays: list[_OverlayText],
    patch_log_dir: Path,
    *,
    required_timeline: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    applied: list[dict[str, str]] = []
    statuses: list[dict[str, Any]] = []
    for idx, patch in enumerate(patches):
        patch_file = str(patch.get("patch_file") or "")
        status: dict[str, Any] = {"patch_ref": patch_file, "timeline_index": patch.get("timeline_index", idx)}
        statuses.append(status)
        tree = patch_trees[idx]
        if tree is None:
            status.update(status="failed", reason="apply_root_absent")
            continue
        # The patch goes into its own recorded tree, so this is per patch rather than fixed for the whole sequence.
        status["target_repo"] = tree["root"]
        overlay = overlays[idx]
        if overlay.reason:
            status.update(status="failed", reason=overlay.reason)
            if required_timeline:
                break
            continue
        patch_path = patch_log_dir / f"{idx:03d}_{Path(patch_file).stem or 'patch'}.diff"
        patch_path.write_text(overlay.content, encoding="utf-8")
        try:
            if tree["use_nogit"]:
                from ._nogit_patch import _apply_patch_no_git

                backups_so_far = tree["nogit_backups"]
                nogit_apply = _apply_patch_no_git(
                    Path(tree["root"]),
                    patch_path,
                    patch_log_dir / "patch_backups",
                    seq_offset=len(backups_so_far),
                )
                if not nogit_apply.ok:
                    raise RuntimeError(nogit_apply.error or "nogit patch apply failed")
                backups_so_far.extend(nogit_apply.backups)
                method = "already_present" if nogit_apply.present_by else "applied_nogit"
            else:
                method = _git_apply_overlay(
                    tree["root"],
                    patch_path,
                    overlay.content,
                    required_timeline=required_timeline,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            log.warning("baseline_executor: warm patch apply failed for %s: %s", patch_file, exc)
            reason = "nogit_apply_failed" if tree["use_nogit"] else "git_apply_failed"
            status.update(status="failed", reason=reason, detail=str(exc)[:500])
            if required_timeline:
                break
            continue
        if method in ("applied", "applied_3way", "applied_nogit"):
            tree["mutated"] = True
        applied.append({"patch_file": patch_file, "status": method})
        status.update(status=method, applied_patch=str(patch_path))
    return applied, statuses


def _warm_replay_report(
    trees: dict[str, dict[str, Any]],
    applied: list[dict[str, str]],
    statuses: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = trees[next(iter(trees))]
    last = statuses[-1] if statuses else {}
    report = {
        "required": True,
        "status": "prepared",
        "patches": statuses,
        "applied": applied,
        "failed_ref": "",
        "pre_sha": primary["pre_sha"],
        "target_repo": primary["root"],
        "snapshot_manifest": primary["snapshot_manifest"],
        "trees": records,
        "rolled_back": False,
    }
    if last.get("status") != "failed":
        return report
    # One overlay failing voids the whole replay, and the sequence may already have written to trees before
    # this one, so every tree is restored rather than just the one that failed.
    restore = _revert_warm_patch_trees(trees.values())
    return {
        **report,
        "status": "failed",
        "failed_ref": last["patch_ref"],
        "failure": last["reason"],
        "rolled_back": bool(restore.get("ok")),
        "rollback": restore,
    }


def _apply_warm_patches(
    params: dict[str, Any],
    output_dir: Path,
    *,
    before_mutation: Any = None,
) -> list[dict[str, str]] | dict[str, Any]:
    """Apply warm-replay code patches, each into the checkout it was taken from."""
    if not (patches := params.get("patches")):
        return []
    required = bool(params.get("required_patch_timeline"))
    # Each overlay applies into the tree its gain was measured on.
    roots = [str(patch.get("framework_root") or "").strip() for patch in patches]
    trees = _warm_patch_trees(roots)
    overlays = _screened_overlays(patches)
    if required and (refusal := _refuse_warm_replay_before_apply(patches, roots, trees, overlays)) is not None:
        return refusal
    if not trees:
        return []
    patch_log_dir = output_dir / "warm_patches"
    patch_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        _snapshot_warm_patch_trees(trees, roots, overlays, output_dir)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        restore = _revert_warm_patch_trees(trees.values())
        if not required:
            log.warning(
                "baseline_executor: skipping legacy warm patches because the "
                "rollback snapshot could not be created: %s",
                exc,
            )
            return []
        primary = trees[next(iter(trees))]
        return {
            **_refused_warm_replay(patches, f"snapshot_failed:{type(exc).__name__}"),
            "pre_sha": primary["pre_sha"],
            "target_repo": primary["root"],
            "trees": _warm_tree_records(trees),
            "rolled_back": bool(restore.get("ok")),
        }
    if any(tree["snapshot_manifest"] for tree in trees.values()):
        params["_warm_patch_trees"] = pending = _warm_tree_records(trees, before_mutation=True)
        if before_mutation is not None and not bool(before_mutation(pending)):
            primary = trees[next(iter(trees))]
            return {
                **_refused_warm_replay(patches, "pending_state_persist_failed"),
                "required": required,
                "pre_sha": primary["pre_sha"],
                "target_repo": primary["root"],
                "snapshot_manifest": primary["snapshot_manifest"],
                "trees": _warm_tree_records(trees),
                "rollback": {"ok": True, "errors": []},
                "rolled_back": True,
            }
    applied, statuses = _apply_warm_patch_sequence(
        patches,
        [trees.get(root) for root in roots],
        overlays,
        patch_log_dir,
        required_timeline=required,
    )
    records = _warm_tree_records(trees)
    if any(tree["snapshot_manifest"] or tree["nogit_backups"] for tree in trees.values()):
        params["_warm_patch_trees"] = records
    # A best-effort timeline reports only the patches that landed, so without
    # this the per-patch reasons computed above would die with this frame --
    # and they are the only record of the ones that did not.
    params["_warm_patch_statuses"] = statuses
    if not required:
        return applied
    return _warm_replay_report(trees, applied, statuses, records)


def _warm_patches_applied(
    patch_application: list[dict[str, str]] | Mapping[str, Any],
) -> list[dict[str, str]]:
    if isinstance(patch_application, Mapping):
        return list(patch_application.get("applied") or [])
    return list(patch_application)


def _stamp_warm_patch_trees(
    result: dict[str, Any],
    patch_application: Mapping[str, Any],
) -> None:
    """Record which trees this round patched, for prelude to promote or restore."""
    trees = list(patch_application.get("trees") or [])
    primary = trees[0] if trees else {}
    result["warm_patch_result"] = dict(patch_application)
    result["warm_patch_trees"] = trees
    result["warm_patch_pre_sha"] = str(primary.get("pre_sha") or "")
    result["warm_patch_target"] = str(primary.get("root") or "")
    result["warm_patch_snapshot_manifest"] = primary.get("snapshot_manifest")


def _stamp_warm_patch_outcome(
    result: dict[str, Any],
    patch_application: list[dict[str, str]] | Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    """Carry the patch application's report into the round's result.

    A required timeline reports the structure prelude promotes or restores
    from. A best-effort one reports only the patches that landed, so its
    per-patch statuses are lifted out of ``params`` instead: a patch that
    silently failed to apply would otherwise leave nothing behind, and the
    round would be measured on a tree nobody downstream can describe.
    """
    if applied := _warm_patches_applied(patch_application):
        result["warm_patches_applied"] = applied
    if isinstance(patch_application, Mapping):
        _stamp_warm_patch_trees(result, patch_application)
        result["warm_kernel_apply_results"] = list(params.get("warm_kernel_apply_results") or [])
        return
    if statuses := list(params.get("_warm_patch_statuses") or []):
        result["warm_patch_result"] = {"required": False, "status": "prepared", "patches": statuses}


def _revert_legacy_warm_patch_trees(params: Mapping[str, Any]) -> dict[str, Any]:
    """Undo a legacy (non-required) apply so nothing leaks into the next task."""
    return _revert_warm_patch_trees(params.get("_warm_patch_trees") or [])


def restore_warm_kernel_snapshots(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore exact kernel target bytes captured before each mutation."""
    errors: list[str] = []
    for snapshot in reversed(snapshots):
        target = Path(str(snapshot.get("target") or ""))
        try:
            if snapshot.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(str(snapshot.get("backup") or "")).read_bytes())
                if snapshot.get("mode") is not None:
                    target.chmod(int(snapshot["mode"]))
            elif target.exists() or target.is_symlink():
                target.unlink()
            if snapshot.get("existed"):
                expected = Path(str(snapshot.get("backup") or "")).read_bytes()
                if not target.is_file() or target.read_bytes() != expected:
                    raise OSError("kernel restore verification failed")
            elif target.exists() or target.is_symlink():
                raise OSError("kernel target still exists after restore")
        except OSError as exc:
            errors.append(f"{target}:{type(exc).__name__}:{exc}")
    return {"ok": not errors, "errors": errors}


def revert_warm_kernel_patches(
    applied: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rollback kernels exactly, reporting every failure."""
    from ._kernel_agent_tool import _maybe_revert_kernel_patch

    errors: list[str] = []
    for apply_result in reversed(applied):
        if not apply_result.get("manifest_path"):
            continue
        try:
            reverted = _maybe_revert_kernel_patch(apply_result)
            if reverted.get("status") != "ok":
                raise RuntimeError(
                    str(
                        reverted.get("error")
                        or reverted.get("reason")
                        or f"kernel revert status={reverted.get('status')}"
                    )
                )
        except Exception as exc:  # isolation boundary: one applied warm-kernel patch revert
            log.warning("warm-kernel KB: revert failed", exc_info=True)
            errors.append(f"{type(exc).__name__}:{exc}")
    if snapshots:
        restored = restore_warm_kernel_snapshots(snapshots)
        errors.extend(restored.get("errors") or [])
    return {"ok": not errors, "errors": errors}


def _rollback_warm_kernel_apply_results(
    results: Any,
    snapshots: Any = None,
) -> dict[str, Any]:
    """Rollback kernel mutations and report whether every restore succeeded."""
    if not isinstance(results, list):
        return {"ok": False, "errors": ["invalid_apply_results"]}

    return revert_warm_kernel_patches(
        [r for r in results if isinstance(r, dict)],
        list(snapshots) if isinstance(snapshots, list) else None,
    )


_LOG_NAMES = ("benchmark_stderr.log", "benchmark_stdout.log", "server.log")


def _iter_log_tails(root: Path, *, max_bytes: int, limit: int = 64) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, tail)`` for each known log under *root*, over at most *limit* files."""
    seen = 0
    try:
        for path in root.rglob("*.log"):
            if path.name not in _LOG_NAMES:
                continue
            seen += 1
            if seen > limit:
                break
            try:
                with path.open("rb") as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    handle.seek(max(0, size - max_bytes))
                    tail = handle.read().decode("utf-8", "replace")
            except OSError:
                continue
            yield path, tail
    except OSError:
        return


@dataclass(frozen=True)
class BenchmarkAttempt:
    """What every benchmark round of one attempt runs with, and the recorder its rounds report to."""

    timeout_sec: int
    override_result_dir: str | None
    resolved_model: str
    materialized_config_path: Path
    inferencex_path: str
    effective_extra_server_args: str
    params: dict[str, Any]
    ctx: RunnerContext
    run_eval_disabled: bool = False
    serving_lease: Any = None
    recorder: Any = None
    run_index: int = 0


class _BenchmarkRound(NamedTuple):
    config_path: Path
    materialized_config_path: Path
    output_dir: Path
    result_dir: Path
    ctx: RunnerContext
    ctx_extra: Mapping[str, Any]
    cmd: list[str]
    env: dict[str, Any]
    framework: str
    watchdog_server_log: str | None
    capture_meta: dict[str, Any]
    timeout_sec: int
    silence_timeout_sec: Any
    session_deadline_sec: Any
    serving_lease: Any
    server_already_ready: bool
    run_eval_disabled: bool


class _BenchProcess(NamedTuple):
    returncode: Any
    stdout: str
    stderr: str
    runtime_sec: float
    started_unix: float
    server_log_text: str = ""
    boot_observation_ref: Any = None
    workspace: Path | None = None


def _boot_failure(finished: _BenchProcess, server_log: Path) -> tuple[str, str] | None:
    death = server_log_death_excerpt(str(server_log))
    error = death if death is not None else (finished.stderr or finished.stdout)[-2000:]
    texts = (finished.server_log_text, finished.stderr, finished.stdout)
    if is_aiter_jit_registry_mismatch(*texts):
        return "aiter_jit_registry_mismatch", error
    if _is_cuda_graph_capture_failure(*texts):
        return "cuda_graph_capture_failed", error
    return ("server_init_dead", death) if death is not None else None


class BenchmarkRunExecutor:
    """One benchmark action: materialize the config, launch the backend, retry, parse the report."""

    benchmark_watchdog = False
    runs_accuracy_eval = True
    default_timeout_sec: int | None = None
    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
    ):
        """Initialize the executor with launch defaults."""
        # Backend-aware interpreter: bypass uses a plain python3, magpie uses the Magpie-importable venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        self.magpie_python = magpie_python or resolve_benchmark_interpreter()
        # None = resolve from $FRAMEWORK at call time; explicit fixture path wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.shared_state = shared_state

    def _resolve_default_config(self) -> Path:
        """Resolve the benchmark YAML the round runs when the task names none."""
        raise NotImplementedError

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        """Pick the per-task workspace dir."""
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, action, ctx.task.task_id)

    def _resolve_shared_state(self, shared_state: Any | None = None) -> Any:
        """Resolve the live SharedState for a session-scoped flag read/write."""
        state = shared_state or self.shared_state
        if state is None:
            from ...state.shared_state import SharedState

            state = SharedState.load_or_init(self.session_dir)
        return state

    def _eager_fallback_armed(self, shared_state: Any | None = None) -> bool:
        """Peek the one-shot eager fallback flag WITHOUT consuming it."""
        state = self._resolve_shared_state(shared_state)
        return bool(getattr(state, "baseline_eager_fallback", False))

    def _consume_eager_fallback(self, shared_state: Any | None = None) -> bool:
        """Consume the one-shot cuda-graph eager fallback flag from SharedState."""
        state = self._resolve_shared_state(shared_state)
        if not getattr(state, "baseline_eager_fallback", False):
            return False
        state.baseline_eager_fallback = False
        state.save(self.session_dir)
        return True

    def _measurement_succeeded(self, measurement: dict[str, Any]) -> bool:
        return measurement.get("valid_measurement") is True

    def _resolve_timeout(self, params: dict[str, Any]) -> float:
        """Benchmark rounds share one cap; other executor purposes keep their budgets."""
        if self.benchmark_watchdog:
            return resolve_benchmark_timeouts()[1]
        return float(params.get("timeout_sec") or self.default_timeout_sec)

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook after YAML materialization, before launch; a returned dict aborts the round with it."""
        return None

    @staticmethod
    def _failure_carries_markers(
        result: dict[str, Any],
        markers: tuple[str, ...],
        context_markers: tuple[str, ...] | None = None,
    ) -> bool:
        """Whether a failed result's error tail, warnings or logs hit a marker."""

        def _hit(text: str) -> bool:
            if not any(m in text for m in markers):
                return False
            return not context_markers or any(m in text for m in context_markers)

        if _hit(str(result.get("error") or "")):
            return True
        for w in result.get("nonfatal_warnings") or []:
            if _hit(str(w)):
                return True
        out_dir = result.get("output_dir")
        if not out_dir:
            return False
        root = Path(out_dir)
        # Double-run: the failure markers may live in the sibling warmup round, so climb to the shared task root to
        # scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False
        return any(_hit(tail) for _, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES))

    @staticmethod
    def _warn_on_cold_start_delta(
        result: dict[str, Any],
        warmup_tput: Any,
    ) -> None:
        """Warn when the measured round climbed far enough above the warm-up to suggest a server still ramping."""
        warm = float(warmup_tput or 0.0)
        measured = float(result.get("output_throughput") or 0.0)
        if measured <= 0 or (delta_pct := incremental_gain_pct(measured, warm)) is None:
            return
        if delta_pct <= _COLD_START_DELTA_WARN_PCT:
            return
        result.setdefault("nonfatal_warnings", [])
        result["nonfatal_warnings"].append("baseline_cold_start_delta_high")
        log.warning(
            "baseline_executor: measured round is %.1f%% above the warm-up round "
            "(%.1f -> %.1f tok/s); the server may still have been ramping, so the "
            "anchor every later gain is graded against may be low",
            delta_pct,
            warm,
            measured,
        )

    @staticmethod
    def _eval_failure_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        """Detect an eval-rooted baseline failure and capture bounded evidence."""

        def _window(text: str) -> str | None:
            for m in _EVAL_FAILURE_MARKERS:
                i = text.find(m)
                if i != -1:
                    return text[max(0, i - 200) : i + len(m) + 400]
            return None

        w = _window(str(result.get("error") or ""))
        if w is not None:
            return True, w
        for warn in result.get("nonfatal_warnings") or []:
            w = _window(str(warn))
            if w is not None:
                return True, w
        out_dir = result.get("output_dir")
        if not out_dir:
            return False, ""
        root = Path(out_dir)
        # Double-run: the failure markers may live in the sibling warmup round, so climb to the shared task root to
        # scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False, ""
        for path, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES):
            window = _window(tail)
            if window is not None:
                return True, f"{path.name}: {window}"
        return False, ""

    @staticmethod
    def _is_eval_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline result was caused by the accuracy eval."""
        return BenchmarkRunExecutor._failure_carries_markers(result, _EVAL_FAILURE_MARKERS)

    @staticmethod
    def _is_server_unreachable_eval_failure(result: dict[str, Any]) -> bool:
        """Whether an eval-rooted failure happened because the server had gone away.

        Evidence of a refused connection says the eval never reached a verdict, so the run carries no information
        about accuracy or about a missing framework capability -- the two things the enablement lane exists to chase.

        Deliberately not routed through ``_failure_carries_markers``: that climbs out of a round directory to the
        shared task root, so a warmup round whose server crashed would decide the classification of a measure round
        that did reach a verdict -- demoting a real accuracy failure out of the enablement lane, the inverse of what
        this check is for. Scan this round's own directory instead. The markers are frequently absent from
        ``result['error']`` -- that is why the eval-rooted check reads logs at all -- and when the eval-rooted verdict
        came from a log, the refusal that caused it is in the same log.
        """
        texts = [str(result.get("error") or "")]
        texts.extend(str(warning) for warning in result.get("nonfatal_warnings") or [])
        if any(marker in text for text in texts for marker in _EVAL_SERVER_UNREACHABLE_MARKERS):
            return True
        out_dir = result.get("output_dir")
        if not out_dir:
            return False
        root = Path(out_dir)
        if not root.is_dir():
            return False
        return any(
            marker in tail
            for _, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES)
            for marker in _EVAL_SERVER_UNREACHABLE_MARKERS
        )

    @staticmethod
    def _is_moe_runner_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline died on the MoE runner backend in use."""
        return BenchmarkRunExecutor._failure_carries_markers(
            result,
            _MOE_RUNNER_MISSING_MARKERS,
            _MOE_SCHEME_CONTEXT_MARKERS,
        )

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the session the context names."""
        from hyperloom.inference_optimizer.session.session_binding import named_session_scope

        # Only a context that names its session binds one.
        with named_session_scope((getattr(ctx, "extra", None) or {}).get("session_dir")):
            return await self._run(ctx)

    async def _run(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the action inside the bound session."""
        return await self._run_retrying(ctx)

    async def _run_pass(
        self,
        ctx: RunnerContext,
        *,
        recorder: Any,
        attempt_reason: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one pass through :meth:`_run_once`, recorded as its own row."""
        index = recorder.begin_run(attempt_reason=attempt_reason) if recorder is not None else 0
        try:
            result = await self._run_once(ctx, recorder=recorder, run_index=index, **kwargs)
        except BaseException as exc:
            if recorder is not None:
                recorder.end_run(
                    run_index=index,
                    result={"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)},
                )
            raise
        if recorder is not None:
            recorder.end_run(run_index=index, result=result)
        return result

    async def _run_retrying(self, ctx: RunnerContext, *, recorder: Any = None) -> dict[str, Any]:
        """Run the Magpie baseline; an eval-rooted failure fails the round, a MoE-runner failure retries once."""
        result = await self._run_pass(ctx, recorder=recorder, attempt_reason=RUN_INITIAL)
        # An eval that never reached a verdict because the server was gone is a broken measurement, not a statement
        # about accuracy or about a missing framework capability. Leaving it in the eval-rooted branch stamps it as an
        # eval-failure contract and hands it to the enablement lane, which then spends rounds looking for a capability
        # gap that the evidence does not support. It stays a failed baseline either way -- nothing is salvaged and the
        # accuracy gate is untouched -- but it is counted as an ordinary baseline failure, so the existing
        # ``_BASELINE_MAX_TOTAL_FAILURES`` backstop ends the run with the cause it actually had.
        if result.get("status") != "succeeded" and self._is_eval_rooted_failure(result):
            if self._is_server_unreachable_eval_failure(result):
                log.warning(
                    "baseline_executor: the accuracy eval failed because the server was unreachable, not because "
                    "the eval reached a verdict; recording an ordinary baseline failure rather than an enablement "
                    "gap."
                )
                result.setdefault("nonfatal_warnings", []).append("eval_failed_server_unreachable")
            else:
                _, evidence = self._eval_failure_evidence(result)
                if self._eval_enablement_active(ctx):
                    from ._accuracy_gate import EVAL_KIND_RUNTIME_FAILURE

                    log.warning(
                        "baseline_executor: eval-rooted failure; routing to "
                        "enablement instead of salvaging (RUN_EVAL stays on)."
                    )
                    self._stamp_eval_failure_contract(
                        ctx, result, kind=EVAL_KIND_RUNTIME_FAILURE, observed_accuracy=None, evidence=evidence
                    )
                    return result
                result.setdefault("nonfatal_warnings", [])
                result["accuracy_source"] = "eval_unavailable"
                if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
                    log.error(
                        "baseline_executor: failure is eval-rooted (InferenceX run_eval aborted the benchmark) on a "
                        "genuine baseline, whose whole purpose is to establish the accuracy reference. Stopping now "
                        "-- fix the accuracy eval (see the benchmark stdout/stderr for the run_eval error)."
                    )
                    result["nonfatal_warnings"].append("eval_failed_no_fallback_baseline_requires_accuracy")
                    self._request_eval_rooted_baseline_stop(ctx, result)
                else:
                    log.error(
                        "baseline_executor: failure is eval-rooted (InferenceX run_eval aborted the benchmark); "
                        "the round fails with no throughput-only retry."
                    )
                return result
        if result.get("status") != "succeeded" and self._is_moe_runner_rooted_failure(result):
            log.warning(
                "baseline_executor: the server died on a MoE runner backend "
                "that has no implementation for this checkpoint's quant "
                "scheme; retrying once without --moe-runner-backend so the "
                "framework picks the backend itself."
            )
            retry = await self._run_pass(
                ctx,
                recorder=recorder,
                attempt_reason=RUN_AFTER_MOE_RUNNER_FAILURE,
                force_drop_moe_runner_backend=True,
            )
            retry.setdefault("nonfatal_warnings", []).append("moe_runner_backend_fallback_dropped_flag")
            result = retry
        self._maybe_stop_on_missing_baseline_accuracy(ctx, result)
        return result

    @staticmethod
    def _accuracy_stop_death_evidence(result: dict[str, Any]) -> str:
        """The server's dying words, for a round that reached the accuracy stop without a reference.

        A round whose server dies *after* the throughput benchmark still reports
        ``succeeded`` -- the numbers were already measured. The accuracy eval then runs,
        per the round script, against a port nobody is listening on: every request is
        refused, no ``results*.json`` is written, and the session stops for a missing
        accuracy reference under a message that blames the baseline setup, which is the
        one thing that was not wrong.

        The marker this returns proves only that the server died somewhere in that log,
        never that it died before the eval or that it is why the reference is missing --
        so callers must report it as evidence to read, not as an established cause.
        """
        path = str(result.get("server_log_path") or "").strip()
        if not path:
            return ""
        # ``server_log_death_excerpt`` already answers ``None`` for a log it cannot read,
        # so an unreadable path is simply "no death on record" -- no guard of our own.
        return server_log_death_excerpt(path) or ""

    def _eval_enablement_active(self, ctx: RunnerContext) -> bool:
        """Whether an eval failure should route into enablement this run."""
        from ._accuracy_gate import eval_enablement_allowed
        from ._multi_node_env import is_multi_node

        extra = getattr(ctx, "extra", None) or {}
        if not eval_enablement_allowed(self._resolve_shared_state(extra.get("shared_state"))):
            return False
        if is_multi_node():
            return False
        return _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {})

    def _stamp_eval_failure_contract(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
        *,
        kind: str,
        observed_accuracy: float | None,
        evidence: str,
    ) -> dict[str, Any]:
        """Mark ``result`` as an eval-rooted baseline failure for enablement."""
        from ._accuracy_gate import (
            BASELINE_EVAL_ACCURACY_FLOOR_KEY,
            BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
            BASELINE_EVAL_EVIDENCE_KEY,
            BASELINE_EVAL_FAILED_KEY,
            BASELINE_EVAL_FAILURE_KIND_KEY,
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            eval_contract_fingerprint,
        )

        params = ctx.task.params or {}
        framework = _requested_framework(params)
        model = params.get("model") or params.get("resolved_model")
        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        result[BASELINE_EVAL_FAILED_KEY] = True
        result[BASELINE_EVAL_FAILURE_KIND_KEY] = kind
        result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] = observed_accuracy
        result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] = floor
        result[BASELINE_EVAL_EVIDENCE_KEY] = (evidence or "")[:4000]
        # Fingerprint derives from the materialized YAML contract fields only — task/metric are result outputs and may
        # be absent on eval crash, so they must not participate in the stable identity.
        result[BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY] = eval_contract_fingerprint(
            config_path=result.get("materialized_config"),
            framework=framework,
            model=model,
        )
        result["eval_origin"] = "eval"
        return result

    def _request_eval_rooted_baseline_stop(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run for an eval-rooted baseline failure, fail-fast path."""
        from ._accuracy_gate import accuracy_meets_floor, request_baseline_accuracy_stop

        params = ctx.task.params or {}
        framework = _requested_framework(params)
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            acc_val = self._apply_salvaged_accuracy(result, salvaged, shared_state)
            # ``accuracy_meets_floor`` already means "finite, strictly positive and >= floor", so floor 0.0 is the
            # legacy "any usable accuracy".
            if accuracy_meets_floor(acc_val, 0.0):
                log.warning(
                    "baseline_executor: eval-rooted baseline failure, but salvaged "
                    "a valid baseline accuracy=%.4f from a sibling attempt (%s); "
                    "not stopping the run",
                    acc_val,
                    salvaged.get("source_file", ""),
                )
                return
            # A measured zero is a broken baseline, not a usable reference: it must still reach the stop below, now
            # with the score on record.
            log.warning(
                "baseline_executor: eval-rooted baseline failure and the sibling "
                "attempt measured accuracy=%.4f (%s); stopping the run",
                acc_val,
                salvaged.get("source_file", ""),
            )
        request_baseline_accuracy_stop(
            shared_state,
            context=f"baseline:{framework or 'unknown'}:eval_aborted",
        )

    def _maybe_stop_on_missing_baseline_accuracy(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run when a genuine baseline produced no accuracy result."""
        params = ctx.task.params
        if not _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params):
            return
        # A failed status must NOT skip straight past the salvage below.
        if result.get("status") != "succeeded" and not self._is_eval_rooted_failure(result):
            return
        acc = result.get("accuracy")
        eval_enablement = self._eval_enablement_active(ctx)
        from ._accuracy_gate import (
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            EVAL_KIND_GENERATION_PATHOLOGY,
            accuracy_meets_floor,
            classify_accuracy_failure,
            eval_probe_summary,
            request_baseline_accuracy_stop,
        )

        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        # Floor 0.0 is the non-enablement "any positive accuracy is usable" rule; ``accuracy_meets_floor`` already
        # means "finite, strictly positive and >= floor", so it rejects zero either way.
        usable_floor = floor if eval_enablement else 0.0
        if accuracy_meets_floor(acc, usable_floor):
            return
        framework = _requested_framework(params)
        shared_state = ctx.extra.get("shared_state") or self.shared_state
        # Session-level salvage (complements #942): the cold-start guard and the coordinator's retries each run in
        # their own ``runs/baseline/<attempt>`` dir. #942 keeps every attempt's eval output inside that attempt's
        # ``$RESULT_DIR``, but the accuracy-stop decision runs on the *deciding* attempt -- whose dir can be empty
        # when a prior sibling attempt already produced a valid ``results*.json``.
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            acc_val = self._apply_salvaged_accuracy(
                result,
                salvaged,
                shared_state,
                expected_handoff=_is_double_run_accuracy_handoff(result, salvaged),
            )
            if accuracy_meets_floor(acc_val, usable_floor):
                return
            # Salvaged, but unusable (zero or still under the floor): that is a real quality signal, so fall through
            # with the observed value rather than reporting it as a missing measurement.
            acc = acc_val

        # An eval that never reached a verdict because the server was unreachable produced no accuracy signal, so
        # neither the stop below nor the enablement routing has anything to act on; stamping either would put an
        # accuracy-flavoured verdict on evidence that says only that the measurement broke. But when the salvage
        # above did find a measured accuracy, there IS a signal -- a real one, from a round that reached a verdict --
        # and returning here would throw it away, leaving a genuinely under-floor model anchoring nothing.
        if (
            salvaged is None
            and result.get("status") != "succeeded"
            and self._is_server_unreachable_eval_failure(result)
        ):
            return

        # Route into enablement instead of stopping: the throughput baseline stays for diagnostics but is blocked from
        # anchoring.
        if eval_enablement:
            kind = classify_accuracy_failure(acc, floor)
            evidence = (
                f"baseline accuracy did not meet floor: accuracy={acc} floor={floor} "
                f"task={result.get('accuracy_task')} metric={result.get('accuracy_metric')} "
                f"source={result.get('accuracy_source')}"
            )
            # A tripped probe means the eval was cut short because the model never stopped generating, not that it
            # answered and got them wrong.
            if probe := result.get("eval_probe"):
                kind = EVAL_KIND_GENERATION_PATHOLOGY
                evidence = f"{evidence}; {eval_probe_summary(probe)}"
            self._stamp_eval_failure_contract(
                ctx,
                result,
                kind=kind or "",
                observed_accuracy=float(acc) if isinstance(acc, (int, float)) else None,
                evidence=evidence,
            )
            log.warning(
                "baseline_executor: accuracy %s below floor %.4f (kind=%s); routing to "
                "enablement instead of stopping the run.",
                acc,
                floor,
                kind,
            )
            return
        # Only a *missing* reference can be explained by a dead server. A measured zero -- including one salvaged
        # from a sibling attempt -- means the eval did run and did write its results, so blaming the death here would
        # state the opposite of what the artifacts show, however dead the server later became.
        context, cause = f"baseline:{framework or 'unknown'}", ""
        if acc is None and (death := self._accuracy_stop_death_evidence(result)):
            log.error(
                "baseline_executor: no accuracy reference, and this round's server.log "
                "records a fatal engine death. The marker carries no ordering against "
                "the eval, so the two are not established to be the same failure -- but "
                "a broken setup is no longer the only suspect, and this is the excerpt "
                "to read before looking at any config:\n%s",
                death,
            )
            context, cause = (
                f"{context}:server_died",
                "no accuracy result, and a fatal engine death on record for this round",
            )
        request_baseline_accuracy_stop(shared_state, context=context, cause=cause)

    def _apply_salvaged_accuracy(
        self,
        result: dict[str, Any],
        salvaged: dict[str, Any],
        shared_state: Any,
        *,
        expected_handoff: bool = False,
    ) -> float:
        """Record a salvaged sibling accuracy, publishing it as the gate reference only when it can serve as one."""
        from ._accuracy_gate import accuracy_meets_floor

        acc_val = float(salvaged["accuracy"])
        result["accuracy"] = acc_val
        result["accuracy_task"] = salvaged.get("task", "gsm8k")
        result["accuracy_metric"] = salvaged.get("metric", "")
        result["accuracy_source"] = salvaged.get("source_file", "")
        if expected_handoff:
            log.info(
                "baseline_executor: cold-start guard — reading accuracy=%.4f from "
                "the warmup round (%s), the only round that measures it",
                acc_val,
                salvaged.get("source_file", ""),
            )
        else:
            log.warning(
                "baseline_executor: this attempt's RESULT_DIR had no accuracy, "
                "but salvaged a measured baseline accuracy=%.4f from a sibling "
                "attempt (%s)",
                acc_val,
                salvaged.get("source_file", ""),
            )
            result.setdefault("nonfatal_warnings", []).append("baseline_accuracy_salvaged_from_sibling_attempt")
        if shared_state is not None and accuracy_meets_floor(acc_val, 0.0):
            shared_state.baseline_accuracy = acc_val
        return acc_val

    def _salvage_sibling_baseline_accuracy(
        self,
        result: dict[str, Any],
        framework: str | None,
    ) -> dict[str, Any] | None:
        """Return a measured accuracy from a sibling baseline attempt, if any."""
        out = result.get("output_dir")
        if not out:
            return None
        runs_root = Path(out).parent  # .../runs/baseline
        if not runs_root.exists():
            return None
        try:
            from ._accuracy_gate import _finite_score, parse_eval_results

            eval_data = parse_eval_results(runs_root, framework=framework)
        except (OSError, ValueError, AttributeError, TypeError):
            log.debug("baseline_executor: sibling-accuracy salvage scan failed", exc_info=True)
            return None
        if _finite_score(eval_data.get("accuracy")) is None:
            return None
        return eval_data

    async def _run_once(
        self,
        ctx: RunnerContext,
        *,
        force_drop_moe_runner_backend: bool = False,
        recorder: Any = None,
        run_index: int = 0,
    ) -> dict[str, Any]:
        """Resolve what this baseline attempt runs with, write its config, and run its rounds."""
        params = ctx.task.params
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")
        shared_state = ctx.extra.get("shared_state") or self.shared_state
        fw, server_args = self._settle_attempt_server_args(
            params,
            shared_state,
            drop_moe_runner_backend=force_drop_moe_runner_backend,
        )
        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Orchestration-supplied script + result_dir overrides.
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        base_extra_envs = dict(params.get("extra_envs") or {})
        if not self.runs_accuracy_eval:
            base_extra_envs["RUN_EVAL"] = "false"
        # Model path: unified resolver (params → $MODEL_PATH → SharedState), then serving-path normalization
        # (HL_MODEL_BASE / HF cache).
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(shared_state, "model_path", "") or ""),
        )
        await _prepare_aiter_serving_so(base_extra_envs)
        try:
            config_path = materialize_benchmark_config(
                config_path,
                output_dir,
                launch=LaunchLayer(
                    extra_server_args=server_args,
                    extra_envs=base_extra_envs,
                    remove_args=params.get("remove_args"),
                    unset_envs=params.get("unset_envs"),
                    args_mode=str(params.get("args_mode") or "append"),
                ),
                target=BenchmarkTarget(
                    model_path=resolved_model,
                    # gpu_type: task.params > $GPU_TYPE (cli.py canonicalizes mi325x->mi300x).
                    gpu_type=str(params.get("gpu_type") or "").strip().lower()
                    or os.environ.get("GPU_TYPE", "").strip().lower(),
                    benchmark_script=override_script,
                ),
                options=MaterializeOptions(
                    # Only a genuine ``baseline`` task may establish/overwrite the quality reference;
                    # ``replay_warm_recipe`` reuses this executor but must compare against the pure baseline reference
                    # rather than redefine it.
                    establish_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
                    drop_moe_runner_backend=force_drop_moe_runner_backend,
                    flydsl_source_dirs=is_truthy(params.get("flydsl_source_dirs")),
                    agentx_mode=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
                    grading=getattr(shared_state, "grading", None),
                ),
            )
        except (FrameworkScriptMismatchError, ServerArgvInvalidError) as exc:
            return {
                "status": "failed",
                "error_class": getattr(exc, "error_class", "framework_script_mismatch"),
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        from .magpie_agentx import activate_runtime

        activate_runtime(config_path, params.get("runtime_override") or {})
        # Report the invocation now, with the config final and the server not
        # yet booted. This frame is the only one that knows the args as a fact:
        # the one-shot eager fallback and the MoE-runner drop have both had
        # their say by here, and after the launch the same answer can only be
        # guessed at by parsing the server's own log back.
        if recorder is not None:
            recorder.record_invocation(
                run_index=run_index,
                invocation=BaselineInvocation(
                    framework_args=server_args,
                    extra_envs=base_extra_envs,
                    config_path=config_path,
                    framework=fw,
                    model_path=resolved_model,
                    args_mode=str(params.get("args_mode") or "append"),
                ),
            )
        return await self._run_attempt_rounds(
            BenchmarkAttempt(
                timeout_sec=self._resolve_timeout(params),
                override_result_dir=override_result_dir,
                resolved_model=resolved_model,
                materialized_config_path=config_path,
                inferencex_path=os.environ.get("INFERENCEX_PATH", "").strip(),
                effective_extra_server_args=server_args,
                params=params,
                ctx=ctx,
                recorder=recorder,
                run_index=run_index,
            ),
            output_dir,
        )

    def _settle_attempt_server_args(
        self,
        params: dict[str, Any],
        shared_state: Any,
        *,
        drop_moe_runner_backend: bool,
    ) -> tuple[str, str]:
        # One-shot cuda-graph eager fallback: a prior baseline armed state.baseline_eager_fallback.
        server_args = str(params.get("extra_server_args") or "")
        fw = _requested_framework(params) or ""
        if not fw and self._eager_fallback_armed(shared_state):
            log.warning(
                "baseline_executor: eager fallback is armed but framework is "
                "unknown; leaving the one-shot armed (not consuming) so a "
                "later baseline with a known framework can apply it",
            )
        elif fw and self._consume_eager_fallback(shared_state):
            server_args = _with_cuda_graph_disabled(server_args, fw)
            log.warning(
                "baseline_executor: retrying with %s after a prior cuda-graph capture failure (framework=%s)",
                _disable_cuda_graph_flag(fw),
                fw,
            )
        if drop_moe_runner_backend:
            server_args = remove_moe_runner_backend_arg(server_args)
        if server_args or "extra_server_args" in params:
            # Keep the task envelope aligned with the materialized runtime so Roofline fingerprints record one-shot
            # eager fallback accurately.
            params["extra_server_args"] = server_args
        return fw, server_args

    async def _run_attempt_rounds(self, attempt: BenchmarkAttempt, output_dir: Path) -> dict[str, Any]:
        params = attempt.params
        config_path = attempt.materialized_config_path
        extra = getattr(attempt.ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        # MLPerf prepares its delegated client off-loop; native AgentX is entirely managed by Magpie.
        agx_err = await asyncio.to_thread(
            prepare_agentx_runtime,
            env=os.environ,
            inferencex_path=attempt.inferencex_path,
            config_path=config_path,
            active=agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, "")),
        )
        if agx_err:
            agentx = {"error_class": AGENTX_PREFLIGHT_ERROR_CLASS, "error": agx_err}
            return {"status": "failed", **agentx, "output_dir": str(output_dir)}
        if (hook_result := self._after_materialize_config(config_path, output_dir)) is not None:
            hook_result.setdefault("materialized_config", str(config_path))
            hook_result.setdefault("output_dir", str(output_dir))
            return hook_result
        # Cold-start "warmup artifact" guard: the freshly-booted server's first benchmark window pays one-time cold
        # costs that inflate later gains into fictitious "improvements".
        lifecycle = _lifecycle.resolve_lifecycle_params(config_path)
        double_run_wanted = double_run_requested(params)
        double_run = double_run_wanted and lifecycle["eligible"]
        defer_accuracy = is_truthy(params.get("defer_accuracy_until_after_measure"))
        if defer_accuracy and double_run:
            # Only the lifecycle path can reuse the hot server for a staged accuracy round.
            _set_materialized_run_eval(config_path, enabled=False)
        # Whether THIS run actually executes lm-eval, read back from the materialized config the subprocess
        # consumes -- the single source of truth.
        attempt = replace(attempt, run_eval_disabled=materialized_run_eval_disabled(config_path))
        # Asked before the lease, because a round that will not be run should not hold a GPU while being refused.
        refused = self._round_refused_before_ignition(attempt, output_dir, double_run=double_run, ctx_extra=extra)
        if refused is not None:
            return refused
        application = self._prepare_round_patches(params, output_dir, shared_state)
        if isinstance(application, Mapping) and application.get("status") == "failed":
            return self._roll_back_failed_patches(application, params, output_dir, shared_state)
        # Ray-managed GPU execution (§12 T1): one held Ray lease (``num_gpus=TP``) spans this baseline's benchmark
        # rounds — a double-run's warmup + measure reuse one persistent server, so both must run under the same lease.
        from ._grid_runner import _num_gpus_for_config
        from ._ray_serving import maybe_serving_lease

        lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        attempt = replace(attempt, serving_lease=lease)
        # Deep-clean any lingering server right before this round actually boots -- after every earlier gate that can
        # still bail out without starting a server (budget, warm-patch failure), so a round that will not run does not
        # pay for a scan it gets no benefit from.
        framework, port = lifecycle["framework"], lifecycle["port"]
        await self._pre_start_cleanup(pid_dir=output_dir, framework=framework, port=port)
        try:
            if double_run:
                guarded = self._run_cold_start_guarded_rounds
                return await guarded(attempt, output_dir, lifecycle, application, defer_accuracy=defer_accuracy)
            if double_run_wanted:
                log.info(
                    "baseline_executor: cold-start double-run not eligible (%s); running single round.",
                    lifecycle["reason"],
                )
            result = await self._run_reported_round(
                attempt, label=ROUND_SINGLE, config_path=config_path, output_dir=output_dir
            )
            _stamp_warm_patch_outcome(result, application, params)
            return result
        finally:
            if double_run:
                # Defensive teardown so no persistent server leaks.
                _lifecycle.teardown_lifecycle_server(pid_dir=output_dir, framework=framework, port=port)
            else:
                _lifecycle.stop_agentx_servers(output_dir)
            # Revert warm-replay patches to prevent state leakage into subsequent tasks that reuse the same
            # InferenceX checkout. A required timeline's tree is promoted by prelude after this returns, so it must
            # stay patched; reverting it here handed prelude a clean tree and silently lost the replay.
            if application and not isinstance(application, Mapping):
                _revert_legacy_warm_patch_trees(params)
            if lease is not None:
                lease.close()

    def _save_warm_replay_pending(self, state: Any, **fields: Any) -> bool:
        state.warm_replay_pending = {**(state.warm_replay_pending or {}), **fields}
        try:
            state.save(self.session_dir)
        except OSError:
            log.warning("baseline_executor: warm replay pending state persist failed", exc_info=True)
            return False
        return True

    def _prepare_round_patches(
        self,
        params: dict[str, Any],
        output_dir: Path,
        shared_state: Any,
    ) -> list[dict[str, str]] | dict[str, Any]:

        def _persist_recipe_snapshot(tree_records: list[dict[str, Any]]) -> bool:
            if shared_state is None:
                return True
            primary = tree_records[0] if tree_records else {}
            return self._save_warm_replay_pending(
                shared_state,
                status="preparing_required_recipe",
                # The trees are the authority; the singular fields describe the primary one so a resumed legacy
                # reader still finds it.
                recipe_patch_trees=tree_records,
                recipe_patch_target=str(primary.get("root") or ""),
                recipe_patch_pre_sha=str(primary.get("pre_sha") or ""),
                recipe_patch_snapshot_manifest=primary.get("snapshot_manifest"),
            )

        # An overlay recording its own root needs no resolved target, so this only refuses when nothing names a tree
        # at all.
        if params.get("patches") and not any(
            str((entry or {}).get("framework_root") or "").strip()
            for entry in params["patches"]
            if isinstance(entry, dict)
        ):
            return {"status": "failed", "failure": "missing_target_repo", "patches": [], "applied": []}
        application = _apply_warm_patches(params, output_dir, before_mutation=_persist_recipe_snapshot)
        if isinstance(application, Mapping):
            if application.get("status") == "failed":
                return application
            if shared_state is not None:
                self._save_warm_replay_pending(
                    shared_state,
                    status="benchmarking",
                    recipe_patch_trees=list(application.get("trees") or []),
                    recipe_patch_target=str(application.get("target_repo") or ""),
                    recipe_patch_pre_sha=str(application.get("pre_sha") or ""),
                    recipe_patch_snapshot_manifest=application.get("snapshot_manifest"),
                    recipe_patch_statuses=list(application.get("patches") or []),
                )
        if applied := _warm_patches_applied(application):
            log.info(
                "baseline_executor: prepared %d warm-replay code patches: %s",
                len(applied),
                [patch["patch_file"] for patch in applied],
            )
        return application

    def _roll_back_failed_patches(
        self,
        patch_application: Mapping[str, Any],
        params: Mapping[str, Any],
        output_dir: Path,
        live_shared_state: Any,
    ) -> dict[str, Any]:
        kernel_rollback = _rollback_warm_kernel_apply_results(
            params.get("warm_kernel_apply_results"),
            params.get("warm_kernel_snapshots"),
        )
        recipe_rollback = patch_application.get("rollback") or {
            "ok": not patch_application.get("snapshot_manifest"),
            "errors": [],
        }
        rollback_errors = [
            *list(recipe_rollback.get("errors") or []),
            *list(kernel_rollback.get("errors") or []),
        ]
        rolled_back = bool(recipe_rollback.get("ok") and kernel_rollback.get("ok"))
        rollback_result = {
            "ok": rolled_back,
            "recipe": recipe_rollback,
            "kernel": kernel_rollback,
            "errors": rollback_errors,
        }
        if live_shared_state is not None:
            if rolled_back:
                # A successful rollback leaves nothing pending, so this replaces the record rather than merging.
                live_shared_state.warm_replay_pending = {}
                self._save_warm_replay_pending(live_shared_state)
            else:
                live_shared_state.set_stop_reason("warm_replay_rollback_failed")
                self._save_warm_replay_pending(
                    live_shared_state,
                    status="rollback_failed",
                    rollback_errors=rollback_errors,
                )
        return {
            "status": "required_patch_failed" if rolled_back else "required_patch_rollback_failed",
            "error_class": "required_patch_failed" if rolled_back else "warm_replay_rollback_failed",
            "error": str(patch_application.get("failure") or "required recipe timeline patch failed"),
            "required_patch_failure": patch_application,
            "failed_patch_ref": patch_application.get("failed_ref"),
            "warm_kernel_rolled_back": bool(kernel_rollback.get("ok")),
            "warm_replay_rollback": rollback_result,
            "workspace": str(output_dir),
        }

    def _round_refused_before_ignition(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        ignitable, evidence = self._round_affordable_before_ignition(double_run=double_run, ctx_extra=ctx_extra)
        if ignitable:
            return None
        stopped_result = _unlaunched_round_result(
            STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
            round_label="baseline round",
            output_dir=output_dir,
            capture_meta={
                "materialized_config": str(attempt.materialized_config_path),
                "run_eval_disabled": bool(attempt.run_eval_disabled),
            },
        )
        stopped_result["budget_shortfall"] = evidence
        if evidence.get("one_more_measurement_sec"):
            log.warning(
                "baseline_executor: this round (%.0fs) and one variant to read "
                "against it (%.0fs) need %.0fs, and %.0fs is left (bound=%s), so "
                "nothing is booted. A baseline no variant can follow is a "
                "denominator with no numerator; the anchor this session already "
                "measured stands.",
                evidence.get("round_sec", 0.0),
                evidence.get("one_more_measurement_sec", 0.0),
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
        else:
            log.warning(
                "baseline_executor: this round needs %.0fs and only %.0fs is left "
                "(bound=%s), so nothing is booted. The anchor this session already "
                "measured stands.",
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
        return stopped_result

    async def _run_guarded_round(
        self,
        attempt: BenchmarkAttempt,
        label: str,
        round_dir: Path,
        *,
        port: Any,
        **config: Any,
    ) -> dict[str, Any]:
        round_cfg = self._write_lifecycle_config(
            attempt.materialized_config_path,
            round_dir,
            cleanup=config.pop("cleanup"),
            port=port,
            run_eval=config.pop("run_eval"),
        )
        return await self._run_reported_round(
            attempt,
            label=label,
            config_path=round_cfg,
            output_dir=round_dir,
            **config,
        )

    async def _run_cold_start_guarded_rounds(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        lifecycle: Mapping[str, Any],
        application: list[dict[str, str]] | dict[str, Any],
        *,
        defer_accuracy: bool,
    ) -> dict[str, Any]:
        params = attempt.params
        warmup_dir = output_dir / _WARMUP_ROUND_DIR
        log.info(
            "baseline_executor: cold-start guard — warmup round (discarded, boots persistent server) in %s",
            warmup_dir,
        )
        # A replayed KB config is promoted onto ``current_best`` and becomes the reference every later measurement
        # in the session is taken against, so it may not be adopted on throughput alone.
        replayed = str(getattr(attempt.ctx.task, "kind", "") or "") == "replay_warm_recipe"
        # A round that re-attaches to the server the warmup left up rather than booting one of its own.
        hot_round = {"port": lifecycle["port"], "server_already_ready": True}
        warmup_eval = True if replayed and not defer_accuracy else None
        # Each warmup starts a fresh benchmark cap; the session deadline never resets.
        warmup_result = await self._run_guarded_round(
            attempt,
            ROUND_WARMUP,
            warmup_dir,
            port=lifecycle["port"],
            cleanup=False,
            run_eval=warmup_eval,
            warmup_round=True,
        )
        if warmup_result.get("status") != "succeeded":
            # Warmup failure almost certainly recurs, so skip the measured round.
            warmup_result.setdefault("nonfatal_warnings", []).append("baseline_warmup_round_failed")
            log.warning(
                "baseline_executor: warmup round failed (error_class=%s); skipping measured round",
                warmup_result.get("error_class"),
            )
            _stamp_warm_patch_outcome(warmup_result, application, params)
            return warmup_result
        warmup_tput = warmup_result.get("output_throughput")
        await report_progress(
            unit="baseline_round",
            label="warmup",
            index=1,
            total=2,
            status="succeeded",
            output_throughput=warmup_tput,
            runtime_sec=warmup_result.get("subprocess_runtime_sec"),
        )
        if not defer_accuracy and (cold := self._cold_anchor_when_hot_pass_unaffordable(warmup_result, attempt)):
            return cold
        measure_dir = output_dir / _MEASURE_ROUND_DIR
        log.info(
            "baseline_executor: cold-start guard — measured baseline round in %s "
            "(warmup tput=%.1f tok/s discarded, reusing hot server)",
            measure_dir,
            warmup_tput or 0.0,
        )
        # Round 2 (measured): re-attach to the hot server (client only).
        result = await self._run_guarded_round(
            attempt, ROUND_MEASURE, measure_dir, cleanup=not defer_accuracy, run_eval=False, **hot_round
        )
        _stamp_warm_patch_outcome(result, application, params)
        if result.get("status") != "succeeded":
            if result.get("error_class") != SESSION_TIME_EXHAUSTED_CLASS:
                return result
            # The gate before this pass admitted it and the run's clock took it anyway -- the pass overran what it
            # was priced at.
            log.warning(
                "baseline_executor: the run's clock stopped the measured round mid-flight, so the warmup "
                "stands as the baseline. It is the cold anchor a single-round baseline would have produced, "
                "and the marker below says so.",
            )
            dropped = {"reason": "measure_round_reaped_by_the_run", "measure_round_error": result.get("error")}
            return _cold_anchor_from_warmup(warmup_result, dropped=dropped)
        self._record_measured_against_warmup(result, warmup_result)
        if defer_accuracy:
            await self._settle_deferred_accuracy(attempt, output_dir, hot_round, result)
        return result

    def _cold_anchor_when_hot_pass_unaffordable(
        self,
        warmup_result: dict[str, Any],
        attempt: BenchmarkAttempt,
    ) -> dict[str, Any] | None:
        affordable, evidence = self._measure_round_affordable(
            warmup_runtime_sec=warmup_result.get("subprocess_runtime_sec"),
            warmup_post_ready_sec=post_ready_sec_without_eval(warmup_result),
            ctx_extra=getattr(attempt.ctx, "extra", None) or {},
        )
        if affordable:
            return None
        if evidence.get("one_more_measurement_sec"):
            why = "a hot pass (%.0fs) and one variant to read against it (%.0fs) need %.0fs" % (
                evidence.get("measure_round_sec", 0.0),
                evidence.get("one_more_measurement_sec", 0.0),
                evidence.get("expected_cost_sec", 0.0),
            )
        else:
            why = "a hot pass needs %.0fs" % (evidence.get("expected_cost_sec", 0.0),)
        log.warning(
            "baseline_executor: %s, and %.0fs is left (bound=%s), so the hot "
            "pass is not run. It would have bought a denominator nothing "
            "could then be compared to, and its own overtime anchor would "
            "have gone unused. Keeping the warmup as the baseline; it is the "
            "cold anchor a single-round baseline would have produced, and "
            "the GPU time it cost is already spent. The marker below says "
            "the figure is cold so the session's later gains can be read "
            "against a known-depressed denominator.",
            why,
            evidence.get("affordable_sec", 0.0),
            evidence.get("bound", ""),
        )
        return _cold_anchor_from_warmup(warmup_result, dropped=evidence)

    def _record_measured_against_warmup(self, result: dict[str, Any], warmup_result: dict[str, Any]) -> None:
        result.setdefault("nonfatal_warnings", []).append("baseline_double_run_discarded_first")
        # The measured pass re-attaches to the server launched by the warmup pass.
        for evidence_field in ("launch_evidence", "launch_evidence_path", "server_log_path"):
            if warmup_result.get(evidence_field):
                result[evidence_field] = warmup_result[evidence_field]
        warmup_evidence = result.get("launch_evidence")
        if isinstance(warmup_evidence, Mapping):
            measured_evidence = dict(warmup_evidence)
            measured_evidence["warm_reuse"] = {
                **dict(warmup_evidence.get("warm_reuse") or {}),
                "reused_ready_server": True,
                "provenance": "warmup_round",
                "source_server_log_path": str(result.get("server_log_path") or ""),
            }
            result["launch_evidence"] = measured_evidence
        warmup_tput = warmup_result.get("output_throughput")
        warmup_runtime = warmup_result.get("subprocess_runtime_sec")
        result["warmup_round_tput"] = warmup_tput
        self._warn_on_cold_start_delta(result, warmup_tput)
        # The Coordinator promotes ``subprocess_runtime_sec`` into the explore soft-kill anchor.
        if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
            result["measure_round_runtime_sec"] = result.get("subprocess_runtime_sec")
            result["subprocess_runtime_sec"] = round(float(warmup_runtime), 2)
            # The split belongs to round 1 for the same reason its wall-clock does: round 2 re-attached, so it
            # has no boot to separate and its own reading says nothing about what booting this workload costs.
            result["post_ready_runtime_sec"] = warmup_result.get("post_ready_runtime_sec")
        hot = result.get("output_throughput") or 0.0
        cold = warmup_tput or 0.0
        log.info(
            "baseline_executor: cold-start guard — measured "
            "baseline=%.1f tok/s (warmup=%.1f tok/s discarded; "
            "artifact would have been +%.0f%%)",
            hot,
            cold,
            ((hot / cold - 1.0) * 100.0) if cold > 0 else 0.0,
        )

    async def _settle_deferred_accuracy(
        self,
        attempt: BenchmarkAttempt,
        output_dir: Path,
        hot_round: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        params = attempt.params
        hot = float(result.get("output_throughput") or 0.0)
        keep_policy = params.get("post_measure_accuracy_keep_policy")
        if keep_policy is None:
            try:
                min_tput = float(params.get("post_measure_accuracy_min_tput", 0.0) or 0.0)
            except (TypeError, ValueError):
                min_tput = 0.0
            run_accuracy = hot >= min_tput
            skipped = {
                "status": "skipped",
                "reason": "throughput_below_threshold",
                "minimum_tput": min_tput,
                "observed_tput": hot,
            }
        else:
            ctx_extra = getattr(attempt.ctx, "extra", None) or {}
            performance = assess_integrate_performance(
                ctx_extra.get("shared_state") or self.shared_state,
                result,
                **keep_policy,
            )
            graded = performance.graded
            run_accuracy = performance.decision == "KEEP"
            skipped = {
                "status": "skipped",
                "reason": (
                    "intvty_regression"
                    if graded.graded_on_intvty and graded.verdict == "REVERT"
                    else "performance_keep_not_eligible"
                ),
                "graded_objective": graded.objective,
                "candidate": graded.candidate,
                "reference": graded.reference,
                "base_tput": keep_policy["base_tput"],
                "gain_pct": performance.gain_pct,
                "stack_incremental_gain_pct": performance.stack_incremental_gain_pct,
                "degrade_reason": graded.degrade_reason,
            }
        if not run_accuracy:
            result["accuracy_stage"] = skipped
            return
        accuracy_result = await self._run_guarded_round(
            replace(attempt, run_eval_disabled=False),
            ROUND_ACCURACY,
            output_dir / _ACCURACY_ROUND_DIR,
            cleanup=True,
            run_eval=True,
            **hot_round,
        )
        result["accuracy_stage"] = {
            "status": accuracy_result.get("status"),
            "error_class": accuracy_result.get("error_class"),
            "workspace": accuracy_result.get("workspace"),
        }
        if accuracy_result.get("status") != "succeeded":
            result.setdefault("nonfatal_warnings", []).append("post_measure_accuracy_failed")
            return
        for key in ("accuracy", "accuracy_task", "accuracy_metric", "accuracy_source"):
            if accuracy_result.get(key) is not None:
                result[key] = accuracy_result[key]

    def _round_affordable_before_ignition(
        self,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a whole round *and a use for it*, before anything boots."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        return self._round_affordable(
            state,
            round_sec=_phase_state.baseline_round_cost_sec(state, double_run=double_run),
        )

    @staticmethod
    def _round_affordable(state: Any, *, round_sec: float | None) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a round costing ``round_sec`` and a use for it."""
        cold_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
        if cold_sec is None or round_sec is None:
            return True, {"reason": "no_measured_round_to_predict_from"}
        use_sec = 0.0
        if _a_use_must_follow_the_round(state):
            # Without the split, a variant is priced at a whole cold round, which is what it is: a boot and a
            # benchmark that pays the compile.
            use_sec = _phase_state.one_more_measurement_sec(state) or cold_sec
        headroom_sec, evidence = _round_headroom_sec(state)
        if headroom_sec is None:
            return True, evidence
        cost = round_sec + use_sec
        priced = {
            "expected_cost_sec": round(cost, 1),
            "round_sec": round(round_sec, 1),
            "one_more_measurement_sec": round(use_sec, 1),
            **evidence,
        }
        return headroom_sec >= cost, priced

    def _measure_round_affordable(
        self,
        *,
        warmup_runtime_sec: Any,
        warmup_post_ready_sec: Any = None,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget covers the measured round *and a use for it*."""
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        headroom_sec, evidence = _round_headroom_sec(state)
        if headroom_sec is None:
            return True, evidence
        warmup_sec = optional_positive_float(warmup_runtime_sec)
        priced_by = "session_hot_pass"
        benchmark_sec = _phase_state.measured_seconds(state, "baseline_warm_runtime_sec")
        if benchmark_sec is None:
            priced_by = "warmup_post_ready"
            benchmark_sec = optional_positive_float(warmup_post_ready_sec)
        if benchmark_sec is None or warmup_sec is None:
            return True, {"reason": "no_measured_benchmark_to_predict_from", **evidence}
        use_sec = 0.0
        if _a_use_must_follow_the_round(state):
            use_sec = _phase_state.one_more_measurement_sec(state) or warmup_sec
        cost = benchmark_sec + use_sec
        priced = {
            "expected_cost_sec": round(cost, 1),
            "priced_by": priced_by,
            "measure_round_sec": round(benchmark_sec, 1),
            "one_more_measurement_sec": round(use_sec, 1),
            **evidence,
        }
        return headroom_sec >= cost, priced

    def _write_lifecycle_config(
        self,
        base_config_path: Path,
        dest_dir: Path,
        *,
        cleanup: bool,
        port: int,
        run_eval: bool | None = None,
    ) -> Path:
        """Render a per-round YAML injecting ``benchmark.server_lifecycle``."""
        with Path(base_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.setdefault("benchmark", {})
        # The rounds of one attempt share the server's pid files in the slot above their own dirs.
        _lifecycle.inject_lifecycle(
            bench,
            cleanup=cleanup,
            pid_dir=dest_dir.parent,
            port=port,
        )
        if run_eval is not None:
            bench.setdefault("envs", {})["RUN_EVAL"] = "true" if run_eval else "false"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = Path(dest_dir) / "baseline_lifecycle.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    async def _pre_start_cleanup(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Remove the pid/json files a previous round left for this framework and port."""
        for p in (lifecycle_pid_file(pid_dir, framework, port), lifecycle_meta_file(pid_dir, framework, port)):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    async def _run_reported_round(
        self,
        attempt: BenchmarkAttempt,
        *,
        label: str,
        config_path: Path,
        output_dir: Path,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Announce a benchmark round before it blocks, run it, and record it."""
        recorder = attempt.recorder
        await report_progress(unit="baseline_round", label=label, status="started")
        started_at = now_iso("seconds")
        started_monotonic = time.monotonic()

        def _record(result: dict[str, Any]) -> None:
            if recorder is None:
                return
            recorder.record_round(
                run_index=attempt.run_index,
                round_=BaselineRound(
                    label=label,
                    started_at=started_at,
                    duration_sec=round(time.monotonic() - started_monotonic, 3),
                    timeout_sec=attempt.timeout_sec,
                    result=result,
                ),
            )

        try:
            result = await self._run_single_benchmark(
                attempt,
                config_path=config_path,
                output_dir=output_dir,
                **overrides,
            )
        except BaseException as exc:
            # A round that raised still happened, and its row is the only place the timeline can say which round the
            # action died in.
            _record({"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)})
            raise
        _record(result)
        return result

    async def _mn_warmup_pass(self, round_: _BenchmarkRound, round_warnings: list[str]) -> dict[str, Any] | None:
        """Run the discarded multi-node client warmup against the restarted server."""
        # A multi-node round is two client passes of the same shape against a server the round did not boot, and the
        # session's measured figure is one of them -- the round's wall-clock is taken after this pass -- so the pair
        # costs twice it.
        state = round_.ctx_extra.get("shared_state") or self.shared_state
        one_pass_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
        affordable, evidence = self._round_affordable(
            state,
            round_sec=None if one_pass_sec is None else one_pass_sec * 2.0,
        )
        if not affordable:
            log.warning(
                "baseline_executor: a multi-node round is two passes needing %.0fs "
                "and %.0fs is left (bound=%s), so neither is launched. The anchor "
                "this session already measured stands.",
                evidence.get("expected_cost_sec", 0.0),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
            )
            refused = _unlaunched_round_result(
                STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
                round_label="multi-node round",
                output_dir=round_.output_dir,
                capture_meta=round_.capture_meta,
            )
            refused["budget_shortfall"] = evidence
            return refused
        warm_dir = round_.output_dir / "mn_warmup"
        started_unix = time.time()
        # The measurement is discarded, but the returncode is not: this pass is a full benchmark round, so a stop here
        # ends the baseline round.
        warm_rc: int | None = None
        try:
            warm_dir.mkdir(parents=True, exist_ok=True)
            warm_cmd = [str(warm_dir) if c == str(round_.output_dir) else c for c in round_.cmd]
            warm_env = dict(round_.env)
            # The pass runs on the server the measured pass reuses, so it must not send the measured prompts.
            apply_benchmark_round_seed(round_.config_path, warm_env, warmup=True)
            warm_env["RESULT_DIR"] = str(warm_dir)
            warm_env["EVAL_RESULT_DIR"] = str(warm_dir / "eval_output")
            warm_env["SERVER_LOG"] = str(warm_dir / "server.log")
            warm_env["GPU_METRICS_CSV"] = str(warm_dir / "gpu_metrics.csv")
            async with heartbeat_while_output_flows(
                unit="baseline_round",
                label="mn_warmup",
            ) as warm_activity:
                warm_proc = await asyncio.to_thread(
                    run_with_session_kill,
                    warm_cmd,
                    env=warm_env,
                    cwd=str(warm_dir),
                    watch=RoundWatch(
                        timeout=round_.timeout_sec,
                        silence_timeout_sec=round_.silence_timeout_sec,
                        server_already_ready=True,
                        server_log_path=_watchdog_server_log_path(warm_dir, round_.framework),
                        session_deadline_sec=round_.session_deadline_sec,
                    ),
                    on_output=warm_activity.note,
                )
            warm_rc = warm_proc.returncode
            log.info("baseline_executor: MN warmup pass done (discarded) rc=%s", warm_rc)
        except subprocess.TimeoutExpired as exc:
            log.warning("baseline_executor: MN warmup pass hit its own hang backstop (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        except (OSError, ValueError) as exc:
            log.warning("baseline_executor: MN warmup pass failed (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        return _stopped_round_result(
            warm_rc,
            round_label="multi-node warmup pass",
            runtime_sec=max(0.0, time.time() - started_unix),
            output_dir=round_.output_dir,
            capture_meta=round_.capture_meta,
        )

    def _preflight_server_argv(self, round_: _BenchmarkRound, attempt: int) -> dict[str, Any] | None:
        """Ask the installed framework's parser about this round's argv.

        Args:
            round_: The round about to launch; its capture_meta is echoed onto a refusal.
            attempt: This attempt's index within the round slot.

        Returns:
            dict | None: A terminal failure result when the argv is refused;
            ``None`` when the round may proceed -- including every unavailable
            verdict, which must never cost a round.
        """
        from .magpie_agentx import native_config

        if native_config(round_.config_path):
            return None
        from ...bringup import (
            ARGV_INVALID,
            argv_invalid_observation,
            check_server_argv,
            write_boot_observation,
        )
        from ...bringup.argv_preflight import OK, PARSED_AFTER_DROP, UNAVAILABLE, ArgvRepair
        from ._server_argv import config_launch_env, config_server_argv, reseal_config_argv

        sealed = config_server_argv(round_.config_path)
        if not sealed.tokenized or not sealed.argv:
            return None

        effective_launch_env = config_launch_env(round_.config_path, round_.env)
        probe_argv = sealed.argv
        if round_.framework.strip().lower() == "vllm" and effective_launch_env.get("PROFILE") == "1":
            probe_argv = (
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                str(round_.output_dir / "torch_trace"),
                *probe_argv,
            )

        # ``_resolve_shared_state`` is typed loosely and callers inject partial
        # doubles, so the round's repair ledger may not be present at all.
        enablement = getattr(self._resolve_shared_state(), "enablement", None)
        spent: list[str] = enablement.argv_repairs if enablement is not None else []
        verdict = check_server_argv(
            framework=round_.framework,
            argv=probe_argv,
            text=sealed.text,
            launch_env=effective_launch_env,
            repair=ArgvRepair(
                digest=sealed.digest,
                spent=spent,
            ),
        )

        if verdict.status == UNAVAILABLE:
            log.info(
                "argv preflight unavailable (%s): %s; the launch remains the only verdict",
                verdict.reason,
                verdict.detail,
            )
            return None
        if verdict.status == OK:
            if verdict.reason == PARSED_AFTER_DROP:
                reseal_config_argv(round_.config_path, verdict.text)
                if verdict.repaired_digest:
                    spent.append(verdict.repaired_digest)
                round_.capture_meta["server_argv_dropped"] = list(verdict.dropped)
            return None

        observation = argv_invalid_observation(verdict, session_dir=self.session_dir)
        round_.capture_meta["boot_observation_path"] = write_boot_observation(
            observation,
            session_dir=self.session_dir,
            output_dir=round_.output_dir,
            attempt=attempt,
        )
        excerpt = observation.excerpt
        log.error(
            "argv preflight: %s refused the server argv (%s); not launching",
            round_.framework or "the framework",
            verdict.reason,
        )
        return {
            "status": "failed",
            "error_class": ARGV_INVALID,
            "error": f"{round_.framework or 'framework'} rejected the server argv ({verdict.reason}): {verdict.detail}",
            "output_dir": str(round_.output_dir),
            "enablement_launch_log": excerpt.text if excerpt is not None else "",
            "server_argv": list(verdict.argv),
            **round_.capture_meta,
        }

    async def _run_single_benchmark(
        self,
        attempt: BenchmarkAttempt,
        *,
        config_path: Path,
        output_dir: Path,
        server_already_ready: bool = False,
        warmup_round: bool = False,
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result; ``warmup_round`` marks a discarded round."""
        from ._multi_node_env import (
            is_multi_node,
            log_mn_banner,
            magpie_remote_env,
            mn_bench_warmup_enabled,
        )
        from .magpie_agentx import bind_source_evidence

        await asyncio.to_thread(bind_source_evidence, config_path, attempt.params.get("patches") or [])
        result_dir = _resolve_result_dir(output_dir, attempt.override_result_dir)
        env = _benchmark_process_env(config_path, output_dir, result_dir, attempt.inferencex_path)
        # Multi-node (--nodes >= 2): inject MAGPIE_RUN_PHASE=client + BENCHMARK_BASE_URL so Magpie skips its server
        # launch and targets the RayJob head.
        env.update(magpie_remote_env())
        apply_benchmark_round_seed(config_path, env, warmup=warmup_round)
        ctx_extra = getattr(attempt.ctx, "extra", None) or {}
        # The session's wall-clock budget, resolved per round rather than once per task: a baseline runs up to three
        # of them, and a warmup that overran has already spent budget the ones after it were counting on.
        session_deadline_sec, _ = session_grid_bounds(ctx_extra.get("shared_state") or self.shared_state)
        silence_timeout_sec = None
        timeout_sec = attempt.timeout_sec
        if self.benchmark_watchdog:
            silence_timeout_sec, timeout_sec = resolve_benchmark_timeouts()
            env["PYTHONUNBUFFERED"] = "1"
            sync_benchmark_timeout(config_path, timeout_sec)
        # The materialized YAML is authoritative for the framework: params and $FRAMEWORK are both optional on a
        # baseline task.
        framework = (
            _config_framework(attempt.materialized_config_path) or (_requested_framework(attempt.params) or "").lower()
        )
        round_ = _BenchmarkRound(
            config_path=config_path,
            materialized_config_path=attempt.materialized_config_path,
            output_dir=output_dir,
            result_dir=result_dir,
            ctx=attempt.ctx,
            ctx_extra=ctx_extra,
            cmd=build_benchmark_command(python_exe=self.magpie_python, config_path=config_path, output_dir=output_dir),
            env=env,
            framework=framework,
            watchdog_server_log=_watchdog_server_log_path(output_dir, framework),
            # Config/eval-contract facts echoed onto every failure result so an eval-rooted failure can be
            # fingerprinted and re-run by enablement.
            capture_meta={
                "materialized_config": str(attempt.materialized_config_path),
                "result_dir": str(result_dir),
                "run_eval_disabled": bool(attempt.run_eval_disabled),
            },
            timeout_sec=timeout_sec,
            silence_timeout_sec=silence_timeout_sec,
            session_deadline_sec=session_deadline_sec,
            serving_lease=attempt.serving_lease,
            server_already_ready=server_already_ready,
            run_eval_disabled=attempt.run_eval_disabled,
        )
        # Multi-node only: restart sglang/vllm per round for a fresh server.
        restart_failure = await self._restart_multi_node_server(
            round_, server_args=attempt.effective_extra_server_args, resolved_model=attempt.resolved_model
        )
        if restart_failure is not None:
            return restart_failure
        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s", round_.cmd, output_dir)
        round_warnings: list[str] = []
        if is_multi_node() and mn_bench_warmup_enabled() and not ctx_extra.get("mn_round_restarted"):
            warmup = await self._mn_warmup_pass(round_, round_warnings)
            if warmup is not None:
                return warmup
        return await self._measured_round_result(round_, round_warnings)

    async def _restart_multi_node_server(
        self,
        round_: _BenchmarkRound,
        *,
        server_args: str,
        resolved_model: str,
    ) -> dict[str, Any] | None:
        from ._multi_node_server_lifecycle import RoundServer, ServerRestartFailed, restart_server_for_round

        if round_.ctx_extra.get("mn_round_restarted"):
            return None
        try:
            # Merge the reference base UNDER the per-task args (last-wins) so a multi-node per-round restart
            # carries the same reference flags the single-node materialized YAML does.
            from hyperloom.inference_optimizer.grid_server_args import merge_server_args
            from ._workload_envs import resolve_reference_launch

            ref_args, ref_envs, _ = resolve_reference_launch()
            # Fold in the operator ``--server-args`` (``INFERENCE_OPTIMIZER_SERVER_ARGS``).
            operator_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
            base = ref_args
            if operator_args:
                base = merge_server_args(base, operator_args) if base else operator_args
            # Base on ``server_args`` (carries the one-shot cuda-graph eager-fallback flag when armed) so the MN
            # per-round restart keeps that fallback too.
            env = {str(key): str(value) for key, value in ref_envs.items()}
            # PD knobs auto-resolved by the helper from $PD_* env, falling back to state.json.
            await restart_server_for_round(
                RoundServer.from_env(
                    merge_server_args(base, server_args) if base else server_args, resolved_model or None
                ),
                extra_env=env or None,
            )
        except ServerRestartFailed as exc:
            return {
                "status": "failed",
                "error_class": "mn_server_restart_failed",
                "error": str(exc),
                "output_dir": str(round_.output_dir),
            }
        return None

    async def _measured_round_result(self, round_: _BenchmarkRound, round_warnings: list[str]) -> dict[str, Any]:
        from ._multi_node_env import is_multi_node

        workspaces_before = snapshot_workspaces(round_.output_dir)
        started_unix = time.time()
        # Anchor the Magpie parent process cwd to the per-task output_dir.
        round_.output_dir.mkdir(parents=True, exist_ok=True)
        # A reused output_dir may still hold a previous attempt's server.log, whose terminal init markers would be
        # read as this attempt's, and the ready stamp beside it would make this attempt's boot look like it never
        # happened.
        server_log = round_.output_dir / "server.log"
        attempt_index = open_bringup_attempt(round_.output_dir)
        clear_server_ready_stamp(str(server_log))
        refusal = self._preflight_server_argv(round_, attempt_index)
        if refusal is not None:
            return refusal
        ready = bool(round_.server_already_ready or round_.ctx_extra.get("server_already_ready") or is_multi_node())
        if self.benchmark_watchdog and not ready:
            from ._aiter_jit import sweep_stale_aiter_locks_if_dead

            if deleted := (await asyncio.to_thread(sweep_stale_aiter_locks_if_dead)).get("deleted"):
                log.warning("baseline_executor: reaped %d orphaned aiter JIT lock(s) before server launch", deleted)
        try:
            finished = await self._launch_benchmark(round_, started_unix, server_ready=ready)
        except subprocess.TimeoutExpired as exc:
            # A reaped timeout carries no wrapper streams; the server log is the whole of the evidence.
            await asyncio.to_thread(self._record_bringup, round_, attempt_index, workspaces_before)
            reaped = {"error_class": "timeout", "error": f"baseline benchmark exceeded {round_.timeout_sec}s: {exc}"}
            return self._reaped_round_failure(
                round_.output_dir, workspaces_before, started_unix, {**reaped, **round_.capture_meta}
            )
        server_log_text = await asyncio.to_thread(
            self._record_bringup, round_, attempt_index, workspaces_before, finished.stderr, finished.stdout
        )
        boot_ref = round_.capture_meta["boot_observation_path"]
        finished = finished._replace(server_log_text=server_log_text, boot_observation_ref=boot_ref)
        stopped = _stopped_round_result(
            finished.returncode,
            round_label="measured round",
            runtime_sec=finished.runtime_sec,
            output_dir=round_.output_dir,
            capture_meta=round_.capture_meta,
        )
        if stopped is not None:
            return stopped
        # Detokenizer-stall watchdog reap: the server came up healthy but went silent for the stall grace window
        # (hung engine / wedged detokenizer).
        if finished.returncode == DETOKENIZER_STALL_RETURNCODE:
            log.warning(
                "baseline_executor: detokenizer-stall watchdog reaped run "
                "(server ready but log went silent); error_class=detokenizer_stall."
            )
            reaped = {
                "error_class": "detokenizer_stall",
                "returncode": finished.returncode,
                "error": _DETOKENIZER_STALL_ERROR,
                "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            }
            return self._reaped_round_failure(
                round_.output_dir, workspaces_before, started_unix, {**reaped, **round_.capture_meta}
            )
        # Read the server log back before the harvest below: harvesting can copy a leaked ``server.log`` into
        # ``output_dir`` and change what this classification sees.
        boot = _boot_failure(finished, server_log)
        workspace = select_run_workspace(round_.output_dir, known_before=workspaces_before)
        return await self._round_result_from_workspace(
            round_, finished._replace(workspace=workspace), boot, round_warnings
        )

    def _record_bringup(
        self,
        round_: _BenchmarkRound,
        attempt_index: int,
        workspaces_before: Any,
        wrapper_stderr: str = "",
        wrapper_stdout: str = "",
    ) -> str:
        from ...bringup import observe_bringup, write_boot_observation

        # Magpie writes the server log into the workspace it creates, not to $SERVER_LOG.
        workspace = select_run_workspace(round_.output_dir, known_before=workspaces_before)
        log_path = workspace / "server.log" if workspace is not None else round_.output_dir / "server.log"
        read = read_bringup_log(log_path)
        verdict = observe_bringup(
            server_log=read.text,
            server_elapsed_sec=server_child_elapsed_sec(read.text),
            wrapper_stderr=wrapper_stderr,
            wrapper_stdout=wrapper_stdout,
            session_dir=self.session_dir,
        )
        round_.capture_meta["boot_observation_path"] = write_boot_observation(
            verdict.observation,
            session_dir=self.session_dir,
            output_dir=round_.output_dir,
            attempt=attempt_index,
        )
        round_.capture_meta["boot_observation_degraded"] = read.degraded
        return read.text

    async def _launch_benchmark(
        self,
        round_: _BenchmarkRound,
        started_unix: float,
        *,
        server_ready: bool,
    ) -> _BenchProcess:
        if round_.serving_lease is not None:
            # Ray-managed GPU execution (§12 T1): run inside the lease's actor (holds num_gpus across this run's
            # rounds).
            from ._ray_backend import SessionKillRound, strip_visible_devices_from_config

            # No liveness callback is possible here: the round runs inside a Ray actor in another process
            # (potentially on another node) and only its final ``(rc, stdout, stderr)`` crosses back, so there is
            # nothing local to call per line of child output.
            returncode, stdout, stderr = await asyncio.to_thread(
                round_.serving_lease.run_session_kill,
                SessionKillRound(
                    build_benchmark_command(
                        python_exe=self.magpie_python,
                        config_path=strip_visible_devices_from_config(round_.config_path),
                        output_dir=round_.output_dir,
                    ),
                    env=round_.env,
                    cwd=str(round_.output_dir),
                    timeout=round_.timeout_sec,
                    silence_timeout_sec=round_.silence_timeout_sec,
                    server_already_ready=server_ready,
                    server_log_path=round_.watchdog_server_log,
                    session_remaining_sec=session_deadline_to_remaining_sec(round_.session_deadline_sec),
                ),
            )
        else:
            async with heartbeat_while_output_flows(unit="baseline_round", label="benchmark") as activity:
                proc = await asyncio.to_thread(
                    run_with_session_kill,
                    round_.cmd,
                    env=round_.env,
                    cwd=str(round_.output_dir),
                    watch=RoundWatch(
                        timeout=round_.timeout_sec,
                        silence_timeout_sec=round_.silence_timeout_sec,
                        server_already_ready=server_ready,
                        server_log_path=round_.watchdog_server_log,
                        session_deadline_sec=round_.session_deadline_sec,
                    ),
                    on_output=activity.note,
                )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        return _BenchProcess(
            returncode,
            stdout or "",
            stderr or "",
            max(0.0, time.time() - started_unix),
            started_unix,
        )

    @staticmethod
    def _reaped_round_failure(
        output_dir: Path,
        workspaces_before: Any,
        started_unix: float,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        destination = select_run_workspace(output_dir, known_before=workspaces_before) or output_dir
        harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=started_unix)
        return {
            "status": "failed",
            "output_dir": str(output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in harvested],
            **fields,
        }

    async def _round_result_from_workspace(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        round_warnings: list[str],
    ) -> dict[str, Any]:
        workspace = finished.workspace
        # Always-on artifact harvest: copy wrapper-side leaks into the task workspace so failure-path diagnostics
        # survive; mtime gating rejects stale prior-run leaks.
        harvested = harvest_leaked_artifacts(
            workspace if workspace is not None else round_.output_dir,
            subprocess_started_unix=finished.started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if workspace is None:
            return self._no_workspace_failure(round_, finished, boot, harvested)
        report_path = workspace / _BENCHMARK_REPORT_NAME
        report = read_json(report_path, None, require_dict=True)
        measurement = extract_benchmark_measurement(
            report,
            config_path=round_.config_path,
            workspace=workspace,
            subprocess_started_unix=finished.started_unix,
        )
        warnings = round_warnings + list(measurement.pop("nonfatal_warnings", []) or [])
        warnings.extend(f"harvested_leaked_artifact:{src}" for src, _ in harvested)
        failure = self._failed_round_result(round_, finished, boot, measurement, warnings)
        if failure is not None:
            return failure
        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": finished.returncode,
            "output_dir": str(round_.output_dir),
            "result_dir": str(round_.result_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Materialized YAML for THIS baseline.
            "materialized_config": str(round_.materialized_config_path),
            # Magpie subprocess wall-clock (success path only).
            "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            # The benchmark's own share of that wall-clock, boot excluded.
            "post_ready_runtime_sec": _round_post_ready_sec(
                round_.watchdog_server_log,
                started_unix=finished.started_unix,
                runtime_sec=finished.runtime_sec,
            ),
            # Authoritative (materialized-config) view of whether the serving lm-eval ran this run.
            "run_eval_disabled": bool(round_.run_eval_disabled),
            # Where this round's boot observation landed, and whether the log it was classified from could be read.
            "boot_observation_path": finished.boot_observation_ref,
            "boot_observation_degraded": round_.capture_meta.get("boot_observation_degraded", ""),
        }
        await asyncio.to_thread(
            _attach_baseline_launch_evidence,
            result,
            config_path=round_.materialized_config_path,
            output_dir=round_.output_dir,
            framework=round_.framework,
        )
        self._stamp_round_quality(round_, result, report, warnings)
        return result

    def _no_workspace_failure(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        harvested: Any,
    ) -> dict[str, Any]:
        failure_extras = {
            "output_dir": str(round_.output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            **round_.capture_meta,
        }
        # Magpie never created a benchmark_* workspace, so the wrapper never wrote server.log.
        captured = redact_secret_values(finished.stderr + finished.stdout)
        if captured.strip():
            try:
                log_file = round_.output_dir / "baseline_stderr.log"
                log_file.write_text(captured, encoding="utf-8")
                failure_extras["stderr_log_path"] = str(log_file)
            except OSError as exc:
                log.warning("baseline_executor: failed to persist stderr log: %s", exc)
        if boot is not None:
            error_class, error = boot
            return {
                "status": "failed",
                "error_class": error_class,
                "returncode": finished.returncode,
                "error": redact_secret_values(error),
                **failure_extras,
            }
        if finished.returncode != 0:
            tail = redact_secret_values((finished.stderr or finished.stdout)[-2000:])
            err_class = _classify_subprocess_error(finished.runtime_sec, tail)
            return {
                "status": "failed",
                "error_class": err_class,
                "returncode": finished.returncode,
                "subprocess_runtime_sec": round(finished.runtime_sec, 2),
                "error": tail,
                "output_signature": nonzero_output_signature(err_class, finished.stdout, finished.stderr),
                **failure_extras,
            }
        return {
            "status": "failed",
            "error_class": "no_workspace",
            "error": "Magpie completed but produced no benchmark_* workspace",
            **failure_extras,
        }

    def _unmeasured_round_failure(
        self,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        measurement: dict[str, Any],
        report_path: Path,
        tail: str,
    ) -> tuple[str, str]:
        """Name a round that left no usable measurement, from the strongest evidence it has."""
        if boot is not None:
            return boot
        if finished.returncode != 0:
            error_class = _classify_subprocess_error(finished.runtime_sec, tail)
            if error_class == SUBPROCESS_NONZERO and served_complete_protocol(measurement):
                # The run's own artifact records every requested request as served, so the exit
                # happened after the workload rather than during it. That is what the exit code
                # cannot say on its own, and it is the difference between a harness that refused
                # the result and a process that died mid-round.
                error_class = MAGPIE_NONZERO_INVALID_MEASUREMENT
            return error_class, tail
        if not report_path.exists():
            return "no_report", f"{_BENCHMARK_REPORT_NAME} missing under {report_path.parent}"
        return "invalid_measurement", "benchmark report did not contain positive throughput and completed requests"

    def _failed_round_result(
        self,
        round_: _BenchmarkRound,
        finished: _BenchProcess,
        boot: tuple[str, str] | None,
        measurement: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        workspace = round_.output_dir if finished.workspace is None else finished.workspace
        report_path = workspace / _BENCHMARK_REPORT_NAME
        tail = (finished.stderr or finished.stdout)[-2000:]
        failed = {
            "status": "failed",
            "returncode": finished.returncode,
            "output_dir": str(round_.output_dir),
            "workspace": str(workspace),
            "report_path": str(report_path) if report_path.exists() else None,
            "reported_success": measurement.get("reported_success"),
            "subprocess_runtime_sec": round(finished.runtime_sec, 2),
            "nonfatal_warnings": warnings,
            **round_.capture_meta,
        }
        if not self._measurement_succeeded(measurement):
            error_class, error = self._unmeasured_round_failure(finished, boot, measurement, report_path, tail)
            return {
                **failed,
                "error_class": error_class,
                "error": redact_secret_values(error),
                "output_signature": nonzero_output_signature(error_class, finished.stdout, finished.stderr),
            }
        if finished.returncode == 0:
            return None
        nonzero_error = redact_secret_values(tail)
        if not served_complete_protocol(measurement):
            return {
                **failed,
                "error_class": "magpie_nonzero_after_valid_measurement",
                "error": nonzero_error,
            }
        # The round served every request the client recorded as requested, so the exit code is not evidence against
        # the measurement. The cause stays on the result rather than only in the log.
        warnings.append(f"nonzero_rc_after_complete_protocol:{finished.returncode}")
        log.warning(
            "baseline_executor: magpie exited %d after serving its whole protocol (%s of %s requests); "
            "keeping the measurement: %s",
            finished.returncode,
            measurement.get("completed_requests"),
            measurement.get("requested_requests"),
            nonzero_error,
        )
        return None

    def _stamp_round_quality(
        self,
        round_: _BenchmarkRound,
        result: dict[str, Any],
        report: dict[str, Any] | None,
        warnings: list[str],
    ) -> None:
        from hyperloom.inference_optimizer import framework_registry

        from ._accuracy_gate import eval_probe_summary, parse_eval_results, read_eval_probe

        # Parse accuracy eval results (GSM8K for serving, or the image-quality gate for scriptable frameworks).
        eval_framework = (report or {}).get("framework") or os.environ.get("FRAMEWORK") or None
        shared_state = round_.ctx_extra.get("shared_state") or self.shared_state
        benchmark_mode = str(getattr(shared_state, _BENCHMARK_MODE_ATTR, "") or "")
        # RUN_EVAL gates ONLY the serving lm-eval GSM8K run.
        if (
            round_.run_eval_disabled
            and not framework_registry.is_scriptable(eval_framework)
            and not is_agentx_mode(benchmark_mode)
        ):
            # Serving RUN_EVAL was off this run (a profile round or a staged warmup round), so lm-eval did not
            # execute and there is no fresh accuracy to read.
            log.info(
                "baseline_executor: RUN_EVAL disabled this run (serving); skipping accuracy parse (no lm-eval executed)"
            )
        else:
            # Search from ``$RESULT_DIR`` so serving runs survive benchmark_lib.sh moving/cleaning
            # ``$EVAL_RESULT_DIR`` and scriptable quality gates still resolve from Magpie's benchmark reports.
            eval_data = parse_eval_results(round_.result_dir, framework=eval_framework, benchmark_mode=benchmark_mode)
            if eval_data.get("accuracy") is not None:
                result["accuracy"] = eval_data["accuracy"]
                result["accuracy_task"] = eval_data.get("task", "gsm8k")
                result["accuracy_metric"] = eval_data.get("metric", "")
                result["accuracy_source"] = eval_data.get("source_file", "")
                log.info("baseline_executor: accuracy=%.4f (%s)", result["accuracy"], result["accuracy_task"])
            else:
                log.warning("baseline_executor: accuracy eval not found: %s", eval_data.get("error", "unknown"))
            # Records why the score is ~0; the score itself is already correct.
            if eval_probe := read_eval_probe(round_.result_dir):
                result["eval_probe"] = eval_probe
                log.warning("baseline_executor: %s", eval_probe_summary(eval_probe))
        log.info(
            "baseline_executor: %s %.1f %s (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            result["output_throughput"],
            framework_registry.throughput_unit(eval_framework),
            result["e2el_mean_ms"] or 0.0,
        )


class BaselineExecutor(BenchmarkRunExecutor):
    """Class form for tests / DI; ``baseline_executor`` is the bare callable."""

    benchmark_watchdog = True

    def _resolve_default_config(self) -> Path:
        """Resolve the benchmark YAML the round runs when the task names none."""
        return default_baseline_config()

    @staticmethod
    def _client_script_from_config(config_path: Path) -> str | None:
        """Name the client script this round will run, or ``None`` if unknown.

        Magpie selects ``<framework>_<runner_type>.sh``. Naming it keeps the
        tokenizer check to the script that matters: the multimodal variants carry
        the same ``--result-dir`` marker with a different client call, and judging
        them would let an unrelated shape veto a workload whose own script is fine.
        """
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None
        bench = (cfg.get("benchmark") if isinstance(cfg, dict) else {}) or {}
        override = str(bench.get("benchmark_script") or "").strip()
        if override:
            return override
        framework = str(bench.get("framework") or "").strip().lower()
        runner = str(bench.get("runner_type") or "").strip().lower()
        if not framework or not runner:
            return None
        return f"{framework}_{runner}.sh"

    @staticmethod
    def _model_from_config(config_path: Path) -> str:
        """Read the benchmark model path out of the materialized config."""
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return ""
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        return str((bench or {}).get("model") or "").strip()

    @staticmethod
    def _inferencex_root_from_config(config_path: Path) -> str:
        """Resolve the InferenceX checkout the subprocess will ``cd`` into."""
        try:
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
            path = str((bench or {}).get("inferencex_path") or "").strip()
        except (OSError, yaml.YAMLError):
            path = ""
        return path or os.environ.get("INFERENCEX_PATH", "").strip()

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook after YAML materialization, before launch."""
        from .magpie_agentx import native_config

        if native_config(config_path):
            return None
        ix_root = self._inferencex_root_from_config(config_path)
        if ix_root:
            if invalid := inferencex_checkout.unusable_reason(ix_root):
                return {"status": "failed", "error_class": "inferencex_checkout_invalid", "error": invalid}
            ensure_benchmark_lib_eval_dest_patched(Path(ix_root))
            ensure_benchmark_lib_eval_start_patched(Path(ix_root))
        # Best-effort: without it the warmup round pre-fills the measured round's prefix cache, which inflates the
        # anchor but never breaks a round.
        ensure_benchmark_serving_seed_patched(Path(ix_root) if ix_root else None)
        # Required custom tokenizers must reach the throughput client as well as eval.
        tok_mode = _client_tokenizer_mode(self._model_from_config(config_path))
        if tok_mode:
            script_name = self._client_script_from_config(config_path)
            if not script_name or not client_tokenizer_hook_installed(script_name, inferencex_dir=ix_root or None):
                msg = (
                    f"the benchmark client script {script_name or '<unresolved>'!r} cannot be told to load the "
                    f"{tok_mode!r} tokenizer: it has no --tokenizer-mode hook "
                    f"(inferencex={ix_root or '<unset>'}). Use a Magpie revision at or above "
                    "1eb3df69de9cb26cb5c8109e43371dd1abb9bbed. This model's client resolves the "
                    "checkpoint through HF AutoConfig, which cannot map its model_type, so it "
                    "would die before issuing a request and the round would be graded a boot "
                    "failure with the server up."
                )
                log.error("baseline_executor: %s", msg)
                return {
                    "status": "failed",
                    "error_class": "client_tokenizer_unpatchable",
                    "error": msg,
                }
        if materialized_run_eval_disabled(config_path):
            return None
        return self._eval_patch_failure(ix_root)

    def _eval_patch_failure(self, ix_root: str) -> dict[str, Any] | None:
        # Target present but unpatchable is a hard stop; target absent is an unrecognized layout, which warns
        # rather than failing every eval run.
        probe_root = Path(ix_root) if ix_root else None
        # Best-effort, unlike the probe below: without it a refused connection ends the round on
        # UnboundLocalError instead of on the connection, which is worse reporting of a round that
        # was already going to fail -- not a reason to refuse to start one.
        ensure_eval_unbound_outputs_patched(probe_root)
        if not ensure_eval_probe_patched(probe_root):
            msg = (
                "the generation-pathology probe is not installed "
                "(utils/evals/patches/lm_eval_sitecustomize.py, inferencex="
                f"{ix_root or '<unset>'}, INFERENCEX_PATH="
                f"{os.environ.get('INFERENCEX_PATH', '') or '<unset>'}). "
                "A model that never emits EOS will run the accuracy eval to "
                "the full max_tokens budget on every sample and consume the "
                "entire baseline timeout."
            )
            if eval_probe_targets_exist(probe_root):
                log.error("baseline_executor: %s", msg)
                return {
                    "status": "failed",
                    "error_class": "eval_probe_unpatchable",
                    "error": msg,
                }
            log.warning("baseline_executor: %s", msg)
        # Fail LOUDLY (never warn-and-continue) when the fatal eval flag cannot be removed AND this run is meant
        # to execute lm-eval: the benchmark is guaranteed to abort in run_lm_eval, and the accuracy gate then
        # stops the whole session.
        try:
            compat_ok = ensure_eval_concurrency_compat(inferencex_dir=ix_root or None)
        except OSError as exc:
            log.error("baseline_executor: eval-concurrency compat patch failed for %s: %s", ix_root, exc)
            compat_ok = False
        if not compat_ok:
            msg = (
                "accuracy eval cannot run: the redundant "
                "'--concurrent-requests' flag could not be removed from the "
                "Magpie benchmark scripts (MAGPIE_PATH="
                f"{os.environ.get('MAGPIE_PATH', '') or '<unset>'}) and/or "
                "InferenceX's run_lm_eval arg parser could not be made to "
                f"tolerate it (inferencex={ix_root or '<unset>'}). "
                "InferenceX resolves eval concurrency from "
                "EVAL_CONCURRENT_REQUESTS (fallback CONC); the flag is "
                "rejected as 'Unknown parameter: --concurrent-requests' and "
                "aborts the benchmark before any results*.json is written. "
                "Fix the run_eval line (or re-run install.sh against the "
                "Magpie tree that is actually imported at run time) — do "
                "NOT work around this with RUN_EVAL=false."
            )
            log.error("baseline_executor: %s", msg)
            return {
                "status": "failed",
                "error_class": "eval_concurrency_flag_unpatchable",
                "error": msg,
            }
        return self._eval_patch_anchors_result(ix_root)

    # Scoped to the line-replacement patches this hook applies.
    _EVAL_HOOK_ANCHORS = ("eval_dest", "eval_start")
    # Of those, the one whose silent absence corrupts the accuracy gate rather than degrading it: without
    # ``eval_dest`` the results file lands in the cwd and no score is ever parsed.
    _EVAL_CRITICAL_ANCHORS = ("eval_dest",)

    def _eval_patch_anchors_result(self, ix_root: str | None) -> dict[str, Any] | None:
        """Fail the launch when an eval-critical patch can no longer be applied.

        The ``ensure_*`` calls above report a miss as ``False`` and no caller
        reads it, so an upstream edit that moves an anchor takes the patch
        offline silently -- the run still looks healthy and only the symptom (no
        score) shows up much later. This is the same failure mode that took the
        probe offline before it was re-homed to a real file. Checked only when
        this run executes lm-eval; anchors that merely degrade something are
        logged, not fatal.

        Scoped to the checkout Magpie will benchmark. Env-wide discovery also
        reaches the InferenceX bundled with the installed Magpie, which a run
        that pins ``benchmark.inferencex_path`` never executes — judging it
        aborts every eval run on rot in a tree nothing here reads.

        Args:
            ix_root: The resolved InferenceX root for this run, if any.

        Returns:
            An early-return failure dict, or ``None`` to proceed.
        """
        rotted = failed_patch_anchors_in(ix_root) if ix_root else failed_patch_anchors(None)
        broken = [s for s in rotted if s.name in self._EVAL_HOOK_ANCHORS]
        if not broken:
            return None
        for status in broken:
            log.error("baseline_executor: InferenceX patch anchor broken — %s", status.describe())
        fatal = [s for s in broken if s.name in self._EVAL_CRITICAL_ANCHORS]
        if not fatal:
            return None
        msg = (
            "accuracy eval cannot run: Hyperloom redirects lm-eval's results file "
            "by matching exact upstream text, and that text is no longer there "
            f"(inferencex={ix_root or '<unset>'}). Broken: "
            + "; ".join(s.describe() for s in fatal)
            + ". Re-anchor the patch in _inferencex_patcher.py against the "
            "checkout in use, or pin INFERENCEX_REF back to a revision it "
            "matches. Continuing would leave every results*.json in the "
            "benchmark's cwd, where the accuracy parser never looks, so the gate "
            "would see no score at all — do NOT work around this with "
            "RUN_EVAL=false."
        )
        log.error("baseline_executor: %s", msg)
        return {
            "status": "failed",
            "error_class": "inferencex_patch_anchor_broken",
            "error": msg,
        }

    async def _run(self, ctx: RunnerContext) -> dict[str, Any]:
        """Open the baseline event, run the action, and close it either way."""
        params = ctx.task.params or {}
        streak, total = self._failure_counters(ctx)
        recorder = make_baseline_recorder(
            self._resolve_sink(ctx),
            BaselineRequest(
                task_id=str(getattr(ctx.task, "task_id", "") or ""),
                task_kind=str(getattr(ctx.task, "kind", "") or ""),
                reason=str(params.get("reason") or ""),
                framework=self._resolve_framework(ctx),
                establishes_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
                params=params,
                failure_streak_before=streak,
                total_failures_before=total,
            ),
            owns_event=not str(params.get(INLINE_EVENT_PARAM) or ""),
        )
        try:
            result = await self._run_retrying(ctx, recorder=recorder)
            shared_state = (getattr(ctx, "extra", None) or {}).get("shared_state")
            is_agentx = agentx_active(benchmark_mode=getattr(shared_state, _BENCHMARK_MODE_ATTR, ""))
            if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params) and is_agentx:
                refuse_cacheless_agentx_baseline(result)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        if recorder is not None:
            recorder.finish(with_output_per_gpu(result, getattr(shared_state, "tp", None)))
        return result

    def _resolve_sink(self, ctx: RunnerContext) -> Any:
        """Decide which event this measurement's rows belong to."""
        from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import baseline_event_id
        from hyperloom.inference_optimizer.breakdown.recorder.event_sink import EventSink
        from hyperloom.inference_optimizer.session.session_binding import session_is_bound

        params = ctx.task.params or {}
        if not session_is_bound():
            log.warning(
                "baseline timeline: no session bound; this measurement's whole event will "
                "be missing from the breakdown. The coordinator binds at startup, so this "
                "means either that never happened or the context did not name a session"
            )
            return None
        inline = str(params.get(INLINE_EVENT_PARAM) or "")
        if inline:
            return EventSink(inline, producer=_RECORDER_PRODUCER)
        state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
        event = baseline_event_id(
            str(getattr(state, "phase", "") or "unphased"),
            int(getattr(state, "macro_cycle", 0) or 0),
        )
        return EventSink(event, producer=_RECORDER_PRODUCER)

    def _failure_counters(self, ctx: RunnerContext) -> tuple[int | None, int | None]:
        """The session's baseline failure counts as this measurement starts.

        Read here, at the dispatch, because the counters are advanced by the
        write-back once this action has returned -- so the event cannot see its
        own effect on them, and the value it can see is the one that says what
        this attempt was dispatched into.

        Args:
            ctx (RunnerContext): The runner context.

        Returns:
            tuple[int | None, int | None]: The consecutive-failure streak and
                the session total, or ``(None, None)`` when no state is bound.
        """
        with suppress(AttributeError, TypeError, ValueError):
            state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
            return (
                int(getattr(state, "baseline_failure_streak", 0) or 0),
                int(getattr(state, "baseline_total_failures", 0) or 0),
            )
        return (None, None)

    def _resolve_framework(self, ctx: RunnerContext) -> str:
        """Name the serving framework this measurement runs against."""
        params = ctx.task.params or {}
        with suppress(AttributeError, TypeError, ValueError):
            state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
            return str(
                params.get("framework") or getattr(state, "framework", "") or os.environ.get("FRAMEWORK", "")
            ).strip()
        return str(params.get("framework") or os.environ.get("FRAMEWORK", "")).strip()


baseline_executor = BaselineExecutor()


__all__ = [
    "BaselineExecutor",
    "BenchmarkAttempt",
    "BenchmarkRunExecutor",
    "baseline_executor",
]
