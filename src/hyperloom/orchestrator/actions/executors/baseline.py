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
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, NamedTuple

import yaml

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import redact_secret_values, scrub_benchmark_process_env
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    ROUND_ACCURACY,
    ROUND_MEASURE,
    ROUND_SINGLE,
    ROUND_WARMUP,
    RUN_AFTER_MOE_RUNNER_FAILURE,
    RUN_INITIAL,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer import inferencex_checkout
from hyperloom.inference_optimizer.grid_server_args import with_capture_disabled
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...loop.sub_agent_runner import RunnerContext
from ...measurement.integrate_performance import assess_integrate_performance
from ...measurement.runtime_findings import persist_runtime_findings, scan_server_log
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows, report_progress
from ...phases import machine_state as _phase_state
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from . import _server_lifecycle as _lifecycle
from ._aiter_jit import is_aiter_jit_registry_mismatch
from ._launch_evidence import build_launch_evidence, persist_launch_evidence

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
    SERVER_DEAD_RETURNCODE,
    clear_server_ready_stamp,
    post_ready_runtime_sec,
    resolve_benchmark_timeouts,
    run_with_session_kill,
    server_log_death_excerpt,
    session_deadline_to_remaining_sec,
)
from ._accuracy_gate import (
    materialized_run_eval_disabled,
)

from ._workload_envs import (
    VLLM_SOURCE_ROOT_ENVS,
    _client_tokenizer_mode,
    _remove_moe_runner_backend_arg,
    FrameworkScriptMismatchError,
    agentx_active,
    default_baseline_config,
    prepare_agentx_runtime,
)
from ._inferencex_patcher import (
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_lib_eval_start_patched,
    ensure_eval_probe_patched,
    ensure_eval_unbound_outputs_patched,
    eval_probe_targets_exist,
    failed_patch_anchors,
    failed_patch_anchors_in,
)
from ._agentx_prefix_cache import refuse_cacheless_agentx_baseline
from ._magpie_patcher import ensure_client_tokenizer_hook, ensure_eval_concurrency_compat
from ._warm_patch_apply import (
    _apply_warm_patches,
    _revert_legacy_warm_patch_trees,
    _rollback_warm_kernel_apply_results,
    _stamp_warm_patch_outcome,
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


log = logging.getLogger(__name__)


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

# Magpie writes the benchmark's own report under this name in the workspace it creates.
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


# Fast-exit arg errors (vLLM/sglang exits in <30s on bad CLI args) should not consume the slow-baseline retry budget.
FAST_EXIT_THRESHOLD_SEC = 30.0
# The error class of a subprocess that exited nonzero for no recognised cause.
_SUBPROCESS_NONZERO = "subprocess_nonzero"
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

# KV-cache OOM: weights loaded but no room left for the KV cache.
_KV_CACHE_OOM_MARKERS = (
    "no gpu memory for the kv cache",
    "leave no gpu memory",
    "raise --mem-fraction-static above",
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


#: Bytes read from each end of a ``server.log`` when observing a bring-up.
_BRINGUP_LOG_EDGE_BYTES = 65_536

#: Cap on milestone lines carried out of the unread middle; the ladder needs
#: one witness per milestone.
_BRINGUP_MIDDLE_MARKER_LINES = 200

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
        text: The decoded head-and-tail text; empty when no log was written and
            empty when the log could not be read.
        degraded: Empty when the read answered for the log's contents;
            :data:`~hyperloom.orchestrator.bringup.DEGRADED_UNREADABLE` when a
            log that exists could not be read back.
    """

    text: str
    degraded: str = ""


def _middle_marker_lines(handle: BinaryIO, *, start: int, stop: int) -> list[str]:
    """Return the milestone-bearing lines in ``[start, stop)`` of an open log."""
    from ...bringup.ladder import PROGRESS_MARKER_SUBSTRINGS

    handle.seek(start)
    remaining = max(0, stop - start)
    kept: list[str] = []
    carry = b""
    while remaining > 0 and len(kept) < _BRINGUP_MIDDLE_MARKER_LINES:
        chunk = handle.read(min(1 << 20, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        lines = (carry + chunk).split(b"\n")
        carry = lines.pop()
        for raw in lines:
            text = raw.decode("utf-8", "replace")
            lowered = text.lower()
            if any(marker in lowered for marker in PROGRESS_MARKER_SUBSTRINGS):
                kept.append(text)
                if len(kept) >= _BRINGUP_MIDDLE_MARKER_LINES:
                    break
    return kept


def read_bringup_log(path: Path, *, edge_bytes: int = _BRINGUP_LOG_EDGE_BYTES) -> BringupLog:
    """Read a server log for bring-up classification: both edges, plus milestones.

    The edges carry the failure excerpt. The ladder needs a witness for every
    milestone the boot passed, and those are not all near one: a build that logs
    per kernel shape pushes the later rungs megabytes in, which would understate
    ``stage_reached`` and make two unequal boots compare as equal. So the middle
    is streamed for milestone lines and they are carried between the edges.

    A log that was never written and a log the mount refuses to serve are
    different answers, and both are answers: an ESTALE or EIO on the session
    mount degrades the observation rather than costing the round.

    Args:
        path: The log file to read.
        edge_bytes: Bytes taken from each end; a log no larger than twice this
            is read whole.

    Returns:
        BringupLog: The decoded text, head first, and the degraded outcome when
        a log that exists could not be read.
    """
    from ...bringup import DEGRADED_UNREADABLE

    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if size <= edge_bytes * 2:
                handle.seek(0)
                return BringupLog(handle.read().decode("utf-8", "replace"))
            handle.seek(0)
            head = handle.read(edge_bytes)
            middle = _middle_marker_lines(handle, start=edge_bytes, stop=size - edge_bytes)
            handle.seek(size - edge_bytes)
            tail = handle.read(edge_bytes)
    except FileNotFoundError:
        return BringupLog("")
    except OSError as exc:
        log.warning("bringup: server log %s could not be read (%s)", path, exc)
        return BringupLog("", DEGRADED_UNREADABLE)
    segments = [head.decode("utf-8", "replace"), *middle, tail.decode("utf-8", "replace")]
    return BringupLog("\n".join(segments))


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
        config_path=config_path,
        actual_server_log=actual_log,
        framework=framework,
        slot=output_dir,
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


def _stopped_round_result(
    stopped: StoppedByTheRun,
    *,
    round_label: str,
    returncode: int | None,
    runtime_sec: float,
    output_dir: Path,
    capture_meta: dict[str, Any],
    started: bool = True,
) -> dict[str, Any]:
    """Build the result for a round the run itself stopped."""
    detail = stopped.interrupted if started else stopped.never_started
    if started:
        log.warning(
            "baseline_executor: %s reaped after %.1fs: %s; error_class=%s.",
            round_label,
            runtime_sec,
            detail,
            stopped.error_class,
        )
    else:
        log.warning(
            "baseline_executor: %s not launched: %s; error_class=%s.",
            round_label,
            detail,
            stopped.error_class,
        )
    return {
        "status": "failed",
        "error_class": stopped.error_class,
        "returncode": returncode,
        "error": detail,
        "subprocess_runtime_sec": round(runtime_sec, 2),
        "output_dir": str(output_dir),
        **capture_meta,
    }


def _round_headroom_sec(state: Any, session_deadline_sec: float | None) -> tuple[float | None, dict[str, Any]]:
    """Seconds this round's budget may still spend, and the numbers behind it."""
    if state is not None:
        usable_sec = _phase_state.session_usable_seconds(state)
        if usable_sec is not None:
            return usable_sec, {"bound": "session_usable", "affordable_sec": round(usable_sec, 1)}
        outside: dict[str, Any] = {"reason": "unbounded_session_budget"}
    else:
        outside = {"reason": "no_session_state"}
    if session_deadline_sec is None:
        return None, outside
    remaining_sec = max(0.0, session_deadline_sec - time.monotonic())
    return remaining_sec, {**outside, "bound": "session_deadline", "affordable_sec": round(remaining_sec, 1)}


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


def _positive_seconds(value: Any) -> float | None:
    """Coerce a duration a round reported to seconds, or ``None`` when it did not."""
    try:
        seconds = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0.0 else None


def _disable_cuda_graph_flag(framework: str) -> str:
    """Return the framework-correct flag that disables cuda-graph capture."""
    return _DISABLE_CUDA_GRAPH_FLAGS.get(
        (framework or "").strip().lower(),
        "--disable-cuda-graph",
    )


def _with_cuda_graph_disabled(extra_server_args: str, framework: str) -> str:
    """Apply the framework-correct disable-cuda-graph flag once (idempotent)."""
    return with_capture_disabled(extra_server_args or "", _disable_cuda_graph_flag(framework))


def _classify_subprocess_error(
    elapsed_sec: float,
    stderr_tail: str,
    returncode: int | None = None,
) -> str:
    """Return 'fast_exit_arg_error' when the subprocess died fast on an arg
    validation error, else 'subprocess_nonzero'.

    Args:
        elapsed_sec: Subprocess wall-clock runtime in seconds.
        stderr_tail: Tail of the subprocess stderr used for marker matching.
        returncode: The subprocess returncode, read for the sentinels that
            name their own cause.

    Returns:
        The named class for a recognised sentinel, ``"fast_exit_arg_error"``
        for a fast exit caused by argument validation, else
        ``"subprocess_nonzero"``.
    """
    tail = (stderr_tail or "").lower()
    # KV-cache OOM can surface long after weight load; match before the fast-exit elapsed gate below.
    if any(m in tail for m in _KV_CACHE_OOM_MARKERS):
        return "kv_cache_oom"
    if elapsed_sec >= FAST_EXIT_THRESHOLD_SEC:
        return _SUBPROCESS_NONZERO
    if any(p.lower() in tail for p in _ARG_ERROR_PATTERNS):
        return "fast_exit_arg_error"
    if "valueerror:" in tail and any(p in tail for p in _ARG_ERROR_CONTEXT_PATTERNS):
        return "fast_exit_arg_error"
    return _SUBPROCESS_NONZERO


_SIGNATURE_VOLATILE_RE = re.compile(r"0x[0-9a-f]+|[0-9a-f]{8,}|\d+")


def nonzero_output_signature(err_class: str, proc_stdout: str | None, proc_stderr: str | None) -> str:
    """Identify a ``subprocess_nonzero`` failure by the process's whole output, ignoring numbers.

    The result's ``error`` is only the last 2000 characters of one stream, and two
    different causes (a HIP OOM, an NCCL timeout) can end in the same engine-death
    traceback that long, or differ only in the stream ``error`` leaves out, so the
    signature is taken from both full, redacted streams instead. Numbers (pids,
    ports, timestamps, durations) and hex ids differ between two attempts that failed
    for the same reason, so they are folded away.

    Args:
        err_class: The class the failure was given; only ``subprocess_nonzero`` gets a
            signature, because other classes carry a fixed sentence that unrelated
            failures share.
        proc_stdout: The process's whole stdout.
        proc_stderr: The process's whole stderr.

    Returns:
        ``"<class>|<sha256>"``, or ``""`` when there is no signature (another class, or
        no output: two failures that said nothing are not known to be the same failure).
    """
    if err_class != _SUBPROCESS_NONZERO:
        return ""
    text = redact_secret_values(f"{proc_stderr or ''}\n{proc_stdout or ''}").strip().lower()
    if not text:
        return ""
    text = " ".join(_SIGNATURE_VOLATILE_RE.sub("#", text).split())
    return f"{err_class}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


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


def _requested_framework(params: dict[str, Any]) -> str | None:
    """The serving framework the task asked for, falling back to the process default."""
    return str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None


# Above this cold-start delta the measured round is unlikely to have settled either: the observed pathological case
# climbed 14,202 -> 19,374 -> 22,425 tok/s across three rounds of one unchanged config, i.e. +36% into round 2 and
# another +16% into round 3.
_COLD_START_DELTA_WARN_PCT = 25.0


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


class _BenchmarkRound(NamedTuple):
    """One benchmark subprocess: where it runs, what it runs, and the facts its result is read against."""

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


class _BenchmarkOutput(NamedTuple):
    """What the benchmark subprocess itself returned."""

    returncode: Any
    stdout: Any
    stderr: Any
    runtime_sec: float


class _BenchmarkOutcome(NamedTuple):
    """What one benchmark subprocess left behind, and what its server log says about it."""

    failure: dict[str, Any] | None
    returncode: Any
    stdout: Any
    stderr: Any
    runtime_sec: float
    started_unix: float
    workspaces_before: Any
    boot_observation_ref: Any
    server_init_dead: bool
    server_init_dead_error: str
    registry_mismatch: Any
    cuda_graph_capture_failed: Any


def _refused_benchmark(failure: dict[str, Any]) -> _BenchmarkOutcome:
    """A round that produced no measurement, carrying the failure its caller reports."""
    return _BenchmarkOutcome(failure, None, "", "", 0.0, 0.0, (), None, False, "", None, None)


def _loaded_benchmark_report(report_path: Path) -> dict[str, Any] | None:
    """The benchmark's own JSON report, or ``None`` when it is absent or unreadable."""
    if not report_path.exists():
        return None
    try:
        with report_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


class _BenchProcess(NamedTuple):
    """What the benchmark subprocess left behind, and what the server did under it."""

    returncode: Any
    stdout: str
    stderr: str
    runtime_sec: float
    server_init_dead: Any
    server_init_dead_error: Any
    registry_mismatch: Any
    cuda_graph_capture_failed: Any
    capture_meta: Any


class _PatchSettlement(NamedTuple):
    """What applying a recipe's warm patches left behind, or the failure that ends the run."""

    applied: Any = ()
    pre_patch_sha: str = ""
    failure: dict[str, Any] | None = None


class _BaselineRunRequest(NamedTuple):
    """How one baseline attempt differs from the plain one."""

    force_drop_moe_runner_backend: bool
    recorder: Any
    run_index: int


class _FrameworkArgs(NamedTuple):
    """The framework one attempt runs, and the server args it runs with."""

    framework: str
    server_args: str


class _ScriptOverrides(NamedTuple):
    """The orchestration-supplied benchmark script and result dir, or the refusal that rejected one."""

    failure: dict[str, Any] | None
    script: Any
    result_dir: Any


class _RoundSetup(NamedTuple):
    """What one baseline attempt runs with, resolved before its benchmark config is written."""

    failure: dict[str, Any] | None
    ctx: RunnerContext
    params: dict[str, Any]
    shared_state: Any
    ctx_extra: Mapping[str, Any]
    framework: str
    server_args: str
    inferencex_path: str
    output_dir: Path
    config_path: Path
    timeout_sec: float
    resolved_model: str
    resolved_gpu: str
    override_script: Any
    override_result_dir: Any
    extra_envs: dict[str, Any]
    defer_accuracy: bool


class _MaterializedRound(NamedTuple):
    """A baseline attempt whose benchmark config is written and whose runtime is activated."""

    failure: dict[str, Any] | None
    setup: _RoundSetup
    config_path: Path
    lifecycle: Mapping[str, Any]
    double_run_wanted: bool
    double_run: bool
    run_eval_disabled: bool


class _RoundPatches(NamedTuple):
    """The warm-replay overlays this attempt's rounds are measured on top of."""

    failure: dict[str, Any] | None
    application: list[dict[str, str]] | dict[str, Any]
    applied: list[dict[str, str]]
    pre_sha: str


class _RoundExecution(NamedTuple):
    """The arguments every round of one attempt shares, and the resources they hold."""

    common: dict[str, Any]
    lease: Any
    patches: _RoundPatches


class _AccuracyStageDecision(NamedTuple):
    """Whether the staged accuracy round earned its GPU time, and what to record when it did not."""

    run: bool
    skipped: dict[str, Any]


def _refused_round(setup: _RoundSetup, failure: dict[str, Any]) -> _MaterializedRound:
    """An attempt that will not be run, carrying the failure its caller reports."""
    return _MaterializedRound(failure, setup, setup.config_path, {}, False, False, False)


class BenchmarkRunExecutor:
    """One benchmark action: materialize the config, launch the backend, retry, parse the report."""

    benchmark_watchdog = False
    runs_accuracy_eval = True
    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
        default_timeout_sec: int | None = None,
        cwd: Path | str | None = None,
    ):
        """Initialize the executor with launch defaults."""
        # Backend-aware interpreter: bypass uses a plain python3, magpie uses the Magpie-importable venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        self.magpie_python = magpie_python or resolve_benchmark_interpreter()
        # None = resolve from $FRAMEWORK at call time; explicit fixture path wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.shared_state = shared_state
        self.default_timeout_sec = default_timeout_sec
        self.cwd = Path(cwd if cwd is not None else tempfile.gettempdir())

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
        try:
            state = self._resolve_shared_state(shared_state)
            if not getattr(state, "baseline_eager_fallback", False):
                return False
            state.baseline_eager_fallback = False
            state.save(self.session_dir)
            return True
        except Exception:
            log.debug(
                "baseline_executor: eager-fallback flag check failed",
                exc_info=True,
            )
            return False

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
    def _record_baseline_convergence(
        result: dict[str, Any],
        warmup_tput: Any,
    ) -> None:
        """Record how steady the baseline anchor actually is."""
        try:
            from hyperloom.orchestrator.measurement.convergence import assess_convergence

            warm = float(warmup_tput or 0.0)
            measured = float(result.get("output_throughput") or 0.0)
            verdict = assess_convergence([warm, measured])
            record: dict[str, Any] = verdict.to_dict()
            if warm > 0 and measured > 0:
                delta_pct = (measured - warm) / warm * 100.0
                record["cold_start_delta_pct"] = round(delta_pct, 2)
                if delta_pct > _COLD_START_DELTA_WARN_PCT:
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
            result["baseline_convergence"] = record
        except Exception:
            log.debug("baseline convergence record failed", exc_info=True)

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
        from hyperloom.inference_optimizer.session.session_binding import bound_session_or_none, session_scope

        # Only a context that names its session binds one.
        named = (getattr(ctx, "extra", None) or {}).get("session_dir")
        with ExitStack() as stack:
            with suppress(OSError, RuntimeError):
                session = Path(named).resolve() if named else None
                if session is not None and bound_session_or_none() != session:
                    stack.enter_context(session_scope(session))
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
                result.setdefault("nonfatal_warnings", [])
                result["nonfatal_warnings"].append("eval_failed_server_unreachable")
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
            retry.setdefault("nonfatal_warnings", [])
            retry["nonfatal_warnings"].append("moe_runner_backend_fallback_dropped_flag")
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

    def _baseline_accuracy_reference_is_due(self, ctx: RunnerContext, result: dict[str, Any]) -> bool:
        """Whether this round is one the run expects an accuracy reference from."""
        if not _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
            return False
        # A failed status must NOT skip straight past the salvage below.
        return result.get("status") == "succeeded" or self._is_eval_rooted_failure(result)

    def _adopt_sibling_baseline_accuracy(
        self,
        result: dict[str, Any],
        salvaged: dict[str, Any],
        shared_state: Any,
    ) -> float:
        """Take a sibling attempt's measured accuracy as this round's, and say where it came from."""
        expected_handoff = _is_double_run_accuracy_handoff(result, salvaged)
        acc_val = self._apply_salvaged_accuracy(
            result,
            salvaged,
            shared_state,
            expected_handoff=expected_handoff,
        )
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
        return acc_val

    def _route_accuracy_failure_to_enablement(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
        acc: Any,
        floor: float,
    ) -> None:
        """Block the baseline from anchoring and hand its shortfall to the enablement lane."""
        from ._accuracy_gate import (
            EVAL_KIND_GENERATION_PATHOLOGY,
            classify_accuracy_failure,
            eval_probe_summary,
        )

        kind = classify_accuracy_failure(acc, floor)
        observed = float(acc) if isinstance(acc, (int, float)) else None
        evidence = (
            f"baseline accuracy did not meet floor: accuracy={acc} floor={floor} "
            f"task={result.get('accuracy_task')} metric={result.get('accuracy_metric')} "
            f"source={result.get('accuracy_source')}"
        )
        # A tripped probe means the eval was cut short because the model never stopped generating, not that it
        # answered and got them wrong.
        probe = result.get("eval_probe")
        if probe:
            kind = EVAL_KIND_GENERATION_PATHOLOGY
            evidence = f"{evidence}; {eval_probe_summary(probe)}"
        self._stamp_eval_failure_contract(ctx, result, kind=kind or "", observed_accuracy=observed, evidence=evidence)
        log.warning(
            "baseline_executor: accuracy %s below floor %.4f (kind=%s); routing to "
            "enablement instead of stopping the run.",
            acc,
            floor,
            kind,
        )

    def _stop_without_accuracy_reference(
        self,
        result: dict[str, Any],
        acc: Any,
        shared_state: Any,
        framework: str | None,
    ) -> None:
        """Stop the run: a genuine baseline reached a verdict with no usable accuracy behind it."""
        from ._accuracy_gate import request_baseline_accuracy_stop

        # Only a *missing* reference can be explained by a dead server. A measured zero -- including one salvaged
        # from a sibling attempt -- means the eval did run and did write its results, so blaming the death here would
        # state the opposite of what the artifacts show, however dead the server later became.
        death = self._accuracy_stop_death_evidence(result) if acc is None else ""
        if death:
            log.error(
                "baseline_executor: no accuracy reference, and this round's server.log "
                "records a fatal engine death. The marker carries no ordering against "
                "the eval, so the two are not established to be the same failure -- but "
                "a broken setup is no longer the only suspect, and this is the excerpt "
                "to read before looking at any config:\n%s",
                death,
            )
        request_baseline_accuracy_stop(
            shared_state,
            context=f"baseline:{framework or 'unknown'}{':server_died' if death else ''}",
            cause="no accuracy result, and a fatal engine death on record for this round" if death else "",
        )

    def _maybe_stop_on_missing_baseline_accuracy(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run when a genuine baseline produced no accuracy result."""
        if not self._baseline_accuracy_reference_is_due(ctx, result):
            return
        from ._accuracy_gate import DEFAULT_ENABLEMENT_ACCURACY_FLOOR, accuracy_meets_floor

        acc = result.get("accuracy")
        eval_enablement = self._eval_enablement_active(ctx)
        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        if eval_enablement:
            if accuracy_meets_floor(acc, floor):
                return  # a usable baseline accuracy at/above the floor exists
        elif acc is not None and float(acc) > 0.0:
            return  # a usable baseline accuracy exists
        params = ctx.task.params or {}
        framework = _requested_framework(params)
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        # Session-level salvage (complements #942): the cold-start guard and the coordinator's retries each run in
        # their own ``runs/baseline/<attempt>`` dir. #942 keeps every attempt's eval output inside that attempt's
        # ``$RESULT_DIR``, but the accuracy-stop decision runs on the *deciding* attempt -- whose dir can be empty
        # when a prior sibling attempt already produced a valid ``results*.json``.
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            # Salvaged but unusable (zero or still under the floor) is a real quality signal, so the observed value
            # carries on below rather than being reported as a missing measurement. Floor 0.0 reproduces the
            # non-enablement "any positive accuracy is usable" rule; ``accuracy_meets_floor`` rejects zero either way.
            acc = self._adopt_sibling_baseline_accuracy(result, salvaged, shared_state)
            if accuracy_meets_floor(acc, floor if eval_enablement else 0.0):
                return

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
            self._route_accuracy_failure_to_enablement(ctx, result, acc, floor)
            return
        self._stop_without_accuracy_reference(result, acc, shared_state, framework)

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
        if not expected_handoff:
            result.setdefault("nonfatal_warnings", [])
            result["nonfatal_warnings"].append("baseline_accuracy_salvaged_from_sibling_attempt")
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
        except Exception:
            log.debug("baseline_executor: sibling-accuracy salvage scan failed", exc_info=True)
            return None
        if _finite_score(eval_data.get("accuracy")) is None:
            return None
        return eval_data

    def _settle_attempt_server_args(
        self,
        params: dict[str, Any],
        shared_state: Any,
        *,
        drop_moe_runner_backend: bool,
    ) -> _FrameworkArgs:
        """Decide the server args this attempt runs with and align the task envelope with them."""
        # One-shot cuda-graph eager fallback: a prior baseline armed state.baseline_eager_fallback.
        server_args = str(params.get("extra_server_args") or "")
        fw = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip()
        if not fw and self._eager_fallback_armed(shared_state):
            log.warning(
                "baseline_executor: eager fallback is armed but framework is "
                "unknown; leaving the one-shot armed (not consuming) so a "
                "later baseline with a known framework can apply it",
            )
        elif fw and self._consume_eager_fallback(shared_state):
            cg_flag = _disable_cuda_graph_flag(fw)
            server_args = _with_cuda_graph_disabled(
                server_args,
                fw,
            )
            log.warning(
                "baseline_executor: retrying with %s after a prior cuda-graph capture failure (framework=%s)",
                cg_flag,
                fw,
            )
        if drop_moe_runner_backend:
            server_args = _remove_moe_runner_backend_arg(server_args)
        if server_args or "extra_server_args" in params:
            # Keep the task envelope aligned with the materialized runtime so Roofline fingerprints record one-shot
            # eager fallback accurately.
            params["extra_server_args"] = server_args
        return _FrameworkArgs(fw, server_args)

    @staticmethod
    def _sanitized_overrides(params: Mapping[str, Any], output_dir: Path) -> _ScriptOverrides:
        """The orchestration-supplied benchmark script and result dir, refused when either is unsafe."""
        try:
            return _ScriptOverrides(
                None,
                sanitize_script_name(params.get("benchmark_script")),
                sanitize_result_dir(params.get("result_dir")),
            )
        except ValueError as exc:
            failure = {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
            return _ScriptOverrides(failure, None, None)

    def _resolve_round_setup(self, ctx: RunnerContext, request: _BaselineRunRequest) -> _RoundSetup:
        """Resolve everything one baseline attempt runs with, short of writing its benchmark config."""
        params = ctx.task.params or {}
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        args = self._settle_attempt_server_args(
            params,
            shared_state,
            drop_moe_runner_backend=request.force_drop_moe_runner_backend,
        )
        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)
        overrides = self._sanitized_overrides(params, output_dir)
        extra_envs = dict(params.get("extra_envs") or {})
        if not self.runs_accuracy_eval:
            extra_envs["RUN_EVAL"] = "false"
        return _RoundSetup(
            overrides.failure,
            ctx,
            params,
            shared_state,
            extra,
            args.framework,
            args.server_args,
            os.environ.get("INFERENCEX_PATH", "").strip(),
            output_dir,
            config_path,
            self._resolve_timeout(params),
            # Model path: unified resolver (params → $MODEL_PATH → SharedState), then serving-path normalization
            # (HL_MODEL_BASE / HF cache).
            resolve_session_model_path(
                params=params,
                state_model_path=str(getattr(shared_state, "model_path", "") or ""),
                for_serving=True,
            ),
            # gpu_type: task.params > $GPU_TYPE (cli.py canonicalizes mi325x->mi300x).
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower(),
            overrides.script,
            overrides.result_dir,
            extra_envs,
            is_truthy(params.get("defer_accuracy_until_after_measure")),
        )

    def _record_round_invocation(self, setup: _RoundSetup, request: _BaselineRunRequest, config_path: Path) -> None:
        """Report the invocation, with the config final and the server not yet booted.

        This frame is the only one that knows the args as a fact: the one-shot eager fallback and the
        MoE-runner drop have both had their say by here, and after the launch the same answer can only be
        guessed at by parsing the server's own log back.
        """
        if request.recorder is None:
            return
        request.recorder.record_invocation(
            run_index=request.run_index,
            framework_args=setup.server_args,
            extra_envs=setup.extra_envs,
            config_path=config_path,
            framework=setup.framework,
            model_path=setup.resolved_model,
            args_mode=str(setup.params.get("args_mode") or "append"),
        )

    async def _round_preflight_failure(self, setup: _RoundSetup, config_path: Path) -> dict[str, Any] | None:
        """Prepare the AgentX runtime and read the eval decision back, reporting whatever refuses the round."""
        # MLPerf prepares its delegated client off-loop; native AgentX is entirely managed by Magpie.
        agx_err = await asyncio.to_thread(
            prepare_agentx_runtime,
            env=os.environ,
            inferencex_path=setup.inferencex_path,
            config_path=config_path,
            active=agentx_active(setup.shared_state),
        )
        if agx_err:
            return {
                "status": "failed",
                "error_class": AGENTX_PREFLIGHT_ERROR_CLASS,
                "error": agx_err,
                "output_dir": str(setup.output_dir),
            }
        # Whether THIS run actually executes lm-eval, read back from the materialized config the subprocess consumes
        # -- the single source of truth.
        hook_result = self._after_materialize_config(config_path, setup.output_dir)
        if hook_result is None:
            return None
        hook_result.setdefault("materialized_config", str(config_path))
        hook_result.setdefault("output_dir", str(setup.output_dir))
        return hook_result

    async def _materialize_round_config(self, request: _BaselineRunRequest, setup: _RoundSetup) -> _MaterializedRound:
        """Write the benchmark config this attempt runs from and settle how many rounds it takes."""
        params = setup.params
        await _prepare_aiter_serving_so(setup.extra_envs)
        try:
            config_path = materialize_benchmark_config(
                setup.config_path,
                setup.output_dir,
                extra_server_args=setup.server_args,
                extra_envs=setup.extra_envs,
                remove_args=params.get("remove_args"),
                unset_envs=params.get("unset_envs"),
                args_mode=str(params.get("args_mode") or "append"),
                model_path=setup.resolved_model,
                gpu_type=setup.resolved_gpu,
                benchmark_script=setup.override_script,
                # Only a genuine ``baseline`` task may establish/overwrite the quality reference;
                # ``replay_warm_recipe`` reuses this executor but must compare against the pure baseline
                # reference rather than redefine it.
                establish_quality_ref=_should_establish_quality_ref(getattr(setup.ctx.task, "kind", ""), params),
                drop_moe_runner_backend=request.force_drop_moe_runner_backend,
                flydsl_source_dirs=is_truthy(params.get("flydsl_source_dirs")),
                agentx_mode=agentx_active(setup.shared_state),
                grading=getattr(setup.shared_state, "grading", None),
            )
        except FrameworkScriptMismatchError as exc:
            # Cross-framework script override: return a structured failure.
            return _refused_round(
                setup,
                {
                    "status": "failed",
                    "error_class": "framework_script_mismatch",
                    "error": str(exc),
                    "output_dir": str(setup.output_dir),
                },
            )
        from .magpie_agentx import activate_runtime

        activate_runtime(config_path, params.get("runtime_override") or {})
        self._record_round_invocation(setup, request, config_path)
        failure = await self._round_preflight_failure(setup, config_path)
        if failure is not None:
            return _refused_round(setup, failure)
        # Cold-start "warmup artifact" guard: the freshly-booted server's first benchmark window pays one-time cold
        # costs that inflate later gains into fictitious "improvements".
        lifecycle = _lifecycle.resolve_lifecycle_params(config_path)
        double_run_wanted = double_run_requested(params)
        double_run = double_run_wanted and lifecycle["eligible"]
        if setup.defer_accuracy and double_run:
            # Only the lifecycle path can reuse the hot server for a staged accuracy round.
            _set_materialized_run_eval(
                config_path,
                enabled=False,
            )
        return _MaterializedRound(
            None,
            setup,
            config_path,
            lifecycle,
            double_run_wanted,
            double_run,
            materialized_run_eval_disabled(config_path),
        )

    def _round_refused_before_ignition(self, prepared: _MaterializedRound) -> dict[str, Any] | None:
        """Refuse a round the run can no longer afford.

        Asked before the lease, because a round that will not be run should not hold a GPU while being refused.
        """
        ignitable, evidence = self._round_affordable_before_ignition(
            double_run=prepared.double_run,
            ctx_extra=prepared.setup.ctx_extra,
        )
        if ignitable:
            return None
        stopped_result = _stopped_round_result(
            STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
            round_label="baseline round",
            returncode=None,
            runtime_sec=0.0,
            output_dir=prepared.setup.output_dir,
            capture_meta={
                "materialized_config": str(prepared.config_path),
                "run_eval_disabled": bool(prepared.run_eval_disabled),
            },
            started=False,
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

    def _prepare_round_patches(self, prepared: _MaterializedRound) -> _RoundPatches:
        """Apply this attempt's warm-replay overlays and settle what they left behind."""
        setup = prepared.setup
        # No single pre-apply sha: each overlay's own tree records its pre_sha, so the singular field is only a
        # fallback for a legacy reader.
        before_apply_sha = ""

        def _persist_recipe_snapshot(tree_records: list[dict[str, Any]]) -> bool:
            if setup.shared_state is None:
                return True
            pending = dict(getattr(setup.shared_state, "warm_replay_pending", {}) or {})
            primary = tree_records[0] if tree_records else {}
            pending.update(
                {
                    "status": "preparing_required_recipe",
                    # The trees are the authority; the singular fields describe the primary one so a resumed legacy
                    # reader still finds it.
                    "recipe_patch_trees": tree_records,
                    "recipe_patch_target": str(primary.get("root") or ""),
                    "recipe_patch_pre_sha": str(primary.get("pre_sha") or before_apply_sha),
                    "recipe_patch_snapshot_manifest": primary.get("snapshot_manifest"),
                }
            )
            setup.shared_state.warm_replay_pending = pending
            try:
                setup.shared_state.save(self.session_dir)
            except Exception:
                log.warning(
                    "combined warm replay recipe snapshot persist failed",
                    exc_info=True,
                )
                return False
            return True

        application = self._apply_recipe_patches(
            setup.params, setup.output_dir, before_mutation=_persist_recipe_snapshot
        )
        settlement = self._settle_patch_application(
            application,
            params=setup.params,
            output_dir=setup.output_dir,
            before_apply_sha=before_apply_sha,
            live_shared_state=setup.shared_state,
        )
        if settlement.failure is not None:
            return _RoundPatches(settlement.failure, application, [], "")
        if settlement.applied:
            log.info(
                "baseline_executor: prepared %d warm-replay code patches: %s",
                len(settlement.applied),
                [p["patch_file"] for p in settlement.applied],
            )
        return _RoundPatches(None, application, settlement.applied, settlement.pre_patch_sha)

    @staticmethod
    def _stamp_round_patches(result: dict[str, Any], patches: _RoundPatches, params: Mapping[str, Any]) -> None:
        """Carry the warm-replay overlays a round was measured on into its result."""
        if patches.applied:
            result["warm_patches_applied"] = list(patches.applied)
        _stamp_warm_patch_outcome(result, patches.application, params, patches.pre_sha)

    async def _run_prepared_round(self, request: _BaselineRunRequest, prepared: _MaterializedRound) -> dict[str, Any]:
        """Hold one GPU lease over every round this attempt takes, and run them under it."""
        patches = self._prepare_round_patches(prepared)
        if patches.failure is not None:
            return patches.failure
        setup = prepared.setup
        # Ray-managed GPU execution (§12 T1): one held Ray lease (``num_gpus=TP``) spans this baseline's benchmark
        # rounds — a double-run's warmup + measure reuse one persistent server, so both must run under the same lease.
        from ._grid_runner import _num_gpus_for_config
        from ._ray_serving import maybe_serving_lease

        lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(prepared.config_path))
        # Deep-clean any lingering server right before this round actually boots -- after every earlier gate that can
        # still bail out without starting a server (budget, warm-patch failure), so a round that will not run does not
        # pay for a scan it gets no benefit from.
        await self._pre_start_cleanup(
            pid_dir=setup.output_dir,
            framework=prepared.lifecycle["framework"],
            port=prepared.lifecycle["port"],
        )
        execution = _RoundExecution(
            {
                "timeout_sec": setup.timeout_sec,
                "override_result_dir": setup.override_result_dir,
                "resolved_model": setup.resolved_model,
                "materialized_config_path": prepared.config_path,
                "inferencex_path": setup.inferencex_path,
                "effective_extra_server_args": setup.server_args,
                "params": setup.params,
                "ctx": setup.ctx,
                "run_eval_disabled": prepared.run_eval_disabled,
                "serving_lease": lease,
            },
            lease,
            patches,
        )
        if prepared.double_run:
            return await self._run_cold_start_guarded_rounds(request, prepared, execution)
        return await self._run_single_round(request, prepared, execution)

    async def _run_single_round(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
    ) -> dict[str, Any]:
        """Run the one measured round an attempt takes when the cold-start guard does not apply."""
        if prepared.double_run_wanted and not prepared.lifecycle["eligible"]:
            log.info(
                "baseline_executor: cold-start double-run not eligible (%s); running single round.",
                prepared.lifecycle["reason"],
            )
        try:
            result = await self._run_reported_round(
                label=ROUND_SINGLE,
                config_path=prepared.config_path,
                output_dir=prepared.setup.output_dir,
                recorder=request.recorder,
                run_index=request.run_index,
                **execution.common,
            )
            self._stamp_round_patches(result, execution.patches, prepared.setup.params)
            return result
        finally:
            # A required timeline's tree is promoted by prelude after this returns, so it must stay patched;
            # reverting here handed prelude a clean tree and silently lost the replay.
            if execution.patches.applied and not isinstance(execution.patches.application, dict):
                _revert_legacy_warm_patch_trees(prepared.setup.params)
            _lifecycle.stop_agentx_servers(prepared.setup.output_dir)
            if execution.lease is not None:
                execution.lease.close()

    async def _run_warmup_round(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
    ) -> dict[str, Any]:
        """Run round 1: boot and run, leaving the server up for the measured round to re-attach to."""
        setup = prepared.setup
        warmup_dir = setup.output_dir / "warmup_round"
        # A replayed KB config is promoted onto ``current_best`` and becomes the reference every later measurement
        # in the session is taken against, so it may not be adopted on throughput alone.
        force_warmup_eval = (
            str(getattr(setup.ctx.task, "kind", "") or "") == "replay_warm_recipe" and not setup.defer_accuracy
        )
        warmup_cfg = self._write_lifecycle_config(
            prepared.config_path,
            warmup_dir,
            cleanup=False,
            pid_dir=setup.output_dir,
            port=prepared.lifecycle["port"],
            run_eval=True if force_warmup_eval else None,
        )
        log.info(
            "baseline_executor: cold-start guard — warmup round (discarded, boots persistent server) in %s",
            warmup_dir,
        )
        # Each warmup starts a fresh benchmark cap; the session deadline never resets.
        return await self._run_reported_round(
            label=ROUND_WARMUP,
            config_path=warmup_cfg,
            output_dir=warmup_dir,
            recorder=request.recorder,
            run_index=request.run_index,
            **execution.common,
        )

    def _warmup_failure_result(
        self,
        warmup_result: dict[str, Any],
        prepared: _MaterializedRound,
        execution: _RoundExecution,
    ) -> dict[str, Any]:
        """Report the failed warmup as the attempt's result; a warmup failure almost certainly recurs."""
        warmup_result.setdefault("nonfatal_warnings", [])
        warmup_result["nonfatal_warnings"].append(
            "baseline_warmup_round_failed",
        )
        log.warning(
            "baseline_executor: warmup round failed (error_class=%s); skipping measured round",
            warmup_result.get("error_class"),
        )
        self._stamp_round_patches(warmup_result, execution.patches, prepared.setup.params)
        return warmup_result

    def _cold_anchor_when_hot_pass_unaffordable(
        self,
        warmup_result: dict[str, Any],
        prepared: _MaterializedRound,
    ) -> dict[str, Any] | None:
        """Keep the warmup as the baseline when the run can no longer afford the hot pass."""
        if prepared.setup.defer_accuracy:
            return None
        affordable, evidence = self._measure_round_affordable(
            warmup_runtime_sec=warmup_result.get("subprocess_runtime_sec"),
            warmup_post_ready_sec=warmup_result.get("post_ready_runtime_sec"),
            ctx_extra=prepared.setup.ctx_extra,
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

    async def _run_measure_round(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
        warmup_tput: Any,
    ) -> dict[str, Any]:
        """Run round 2: re-attach to the hot server as a client only, and measure it."""
        setup = prepared.setup
        measure_dir = setup.output_dir / "measure_round"
        measure_cfg = self._write_lifecycle_config(
            prepared.config_path,
            measure_dir,
            cleanup=not setup.defer_accuracy,
            pid_dir=setup.output_dir,
            port=prepared.lifecycle["port"],
            run_eval=False,
        )
        log.info(
            "baseline_executor: cold-start guard — measured baseline "
            "round in %s (warmup tput=%.1f tok/s discarded, reusing "
            "hot server)",
            measure_dir,
            warmup_tput or 0.0,
        )
        return await self._run_reported_round(
            label=ROUND_MEASURE,
            server_already_ready=True,
            config_path=measure_cfg,
            output_dir=measure_dir,
            recorder=request.recorder,
            run_index=request.run_index,
            **execution.common,
        )

    def _record_measured_against_warmup(self, result: dict[str, Any], warmup_result: dict[str, Any]) -> None:
        """Credit the measured pass with the boot it re-attached to, and say what the cold figure would have claimed."""
        result.setdefault("nonfatal_warnings", [])
        result["nonfatal_warnings"].append(
            "baseline_double_run_discarded_first",
        )
        # The measured pass re-attaches to the server launched by the warmup pass.
        for evidence_field in (
            "launch_evidence",
            "launch_evidence_path",
            "server_log_path",
        ):
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
        self._record_baseline_convergence(result, warmup_tput)
        # The Coordinator promotes ``subprocess_runtime_sec`` into the explore soft-kill anchor.
        if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
            result["measure_round_runtime_sec"] = result.get(
                "subprocess_runtime_sec",
            )
            result["subprocess_runtime_sec"] = round(
                float(warmup_runtime),
                2,
            )
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

    def _post_measure_accuracy_decision(self, setup: _RoundSetup, result: dict[str, Any]) -> _AccuracyStageDecision:
        """Decide whether this measured round earned the staged accuracy pass."""
        params = setup.params
        hot = float(result.get("output_throughput") or 0.0)
        keep_policy = params.get("post_measure_accuracy_keep_policy")
        if keep_policy is None:
            try:
                min_tput = float(params.get("post_measure_accuracy_min_tput", 0.0) or 0.0)
            except (TypeError, ValueError):
                min_tput = 0.0
            return _AccuracyStageDecision(
                hot >= min_tput,
                {
                    "status": "skipped",
                    "reason": "throughput_below_threshold",
                    "minimum_tput": min_tput,
                    "observed_tput": hot,
                },
            )
        performance = assess_integrate_performance(setup.shared_state, result, **keep_policy)
        graded = performance.graded
        return _AccuracyStageDecision(
            performance.decision == "KEEP",
            {
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
            },
        )

    async def _run_accuracy_round(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
    ) -> dict[str, Any]:
        """Run the staged accuracy round on the server the measured pass left hot."""
        setup = prepared.setup
        accuracy_dir = setup.output_dir / "accuracy_round"
        accuracy_cfg = self._write_lifecycle_config(
            prepared.config_path,
            accuracy_dir,
            cleanup=True,
            pid_dir=setup.output_dir,
            port=prepared.lifecycle["port"],
            run_eval=True,
        )
        return await self._run_reported_round(
            label=ROUND_ACCURACY,
            server_already_ready=True,
            config_path=accuracy_cfg,
            output_dir=accuracy_dir,
            recorder=request.recorder,
            run_index=request.run_index,
            **{
                **execution.common,
                "run_eval_disabled": False,
            },
        )

    async def _settle_deferred_accuracy(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
        result: dict[str, Any],
    ) -> None:
        """Stage the accuracy round after the measurement, or record why it was not worth its GPU time."""
        decision = self._post_measure_accuracy_decision(prepared.setup, result)
        if not decision.run:
            result["accuracy_stage"] = decision.skipped
            return
        accuracy_result = await self._run_accuracy_round(request, prepared, execution)
        result["accuracy_stage"] = {
            "status": accuracy_result.get("status"),
            "error_class": accuracy_result.get("error_class"),
            "workspace": accuracy_result.get("workspace"),
        }
        if accuracy_result.get("status") != "succeeded":
            result.setdefault("nonfatal_warnings", [])
            result["nonfatal_warnings"].append("post_measure_accuracy_failed")
            return
        for key in (
            "accuracy",
            "accuracy_task",
            "accuracy_metric",
            "accuracy_source",
        ):
            if accuracy_result.get(key) is not None:
                result[key] = accuracy_result[key]

    async def _run_cold_start_guarded_rounds(
        self,
        request: _BaselineRunRequest,
        prepared: _MaterializedRound,
        execution: _RoundExecution,
    ) -> dict[str, Any]:
        """Boot on a discarded warmup round, then measure on the hot server it leaves running."""
        setup = prepared.setup
        try:
            warmup_result = await self._run_warmup_round(request, prepared, execution)
            if warmup_result.get("status") != "succeeded":
                return self._warmup_failure_result(warmup_result, prepared, execution)
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
            cold_anchor = self._cold_anchor_when_hot_pass_unaffordable(warmup_result, prepared)
            if cold_anchor is not None:
                return cold_anchor
            result = await self._run_measure_round(request, prepared, execution, warmup_tput)
            self._stamp_round_patches(result, execution.patches, setup.params)
            if result.get("status") != "succeeded" and result.get("error_class") == SESSION_TIME_EXHAUSTED_CLASS:
                # The gate before this pass admitted it and the run's clock took it anyway -- the pass overran what it
                # was priced at.
                log.warning(
                    "baseline_executor: the run's clock stopped the measured "
                    "round mid-flight, so the warmup stands as the baseline. It "
                    "is the cold anchor a single-round baseline would have "
                    "produced, and the marker below says so.",
                )
                return _cold_anchor_from_warmup(
                    warmup_result,
                    dropped={
                        "reason": "measure_round_reaped_by_the_run",
                        "measure_round_error": result.get("error"),
                    },
                )
            if result.get("status") == "succeeded":
                self._record_measured_against_warmup(result, warmup_result)
                if setup.defer_accuracy:
                    await self._settle_deferred_accuracy(request, prepared, execution, result)
            return result
        finally:
            # Defensive teardown so no persistent server leaks.
            _lifecycle.teardown_lifecycle_server(
                pid_dir=setup.output_dir,
                framework=prepared.lifecycle["framework"],
                port=prepared.lifecycle["port"],
            )
            # Revert warm-replay patches to prevent state leakage into subsequent tasks that reuse the same InferenceX
            # checkout.
            if execution.patches.applied and not isinstance(execution.patches.application, dict):
                _revert_legacy_warm_patch_trees(setup.params)
            if execution.lease is not None:
                execution.lease.close()

    async def _run_once(
        self,
        ctx: RunnerContext,
        *,
        force_drop_moe_runner_backend: bool = False,
        recorder: Any = None,
        run_index: int = 0,
    ) -> dict[str, Any]:
        """Run the Magpie baseline benchmark and parse its result."""
        request = _BaselineRunRequest(force_drop_moe_runner_backend, recorder, run_index)
        setup = self._resolve_round_setup(ctx, request)
        if setup.failure is not None:
            return setup.failure
        prepared = await self._materialize_round_config(request, setup)
        if prepared.failure is not None:
            return prepared.failure
        refused = self._round_refused_before_ignition(prepared)
        if refused is not None:
            return refused
        return await self._run_prepared_round(request, prepared)

    def _settle_patch_application(
        self,
        patch_application: Any,
        *,
        params: Mapping[str, Any],
        output_dir: Any,
        before_apply_sha: str,
        live_shared_state: Any,
    ) -> _PatchSettlement:
        """Read what applying the recipe's warm patches left behind.

        A required timeline patch that failed is rolled back -- the kernel half
        first, then the recipe half -- and reported as the failure that ends the
        run, because a baseline measured over a half-applied tree scores nothing.
        """
        applied: Any = []
        pre_patch_sha = ""
        if isinstance(patch_application, dict):
            applied = list(patch_application.get("applied") or [])
            pre_patch_sha = str(patch_application.get("pre_sha") or "")
            if patch_application.get("status") == "failed":
                return self._roll_back_failed_patches(
                    patch_application,
                    params=params,
                    output_dir=output_dir,
                    live_shared_state=live_shared_state,
                )
            if live_shared_state is not None:
                pending = dict(getattr(live_shared_state, "warm_replay_pending", {}) or {})
                pending.update(
                    {
                        "status": "benchmarking",
                        "recipe_patch_trees": list(patch_application.get("trees") or []),
                        "recipe_patch_target": str(patch_application.get("target_repo") or ""),
                        "recipe_patch_pre_sha": pre_patch_sha,
                        "recipe_patch_snapshot_manifest": patch_application.get("snapshot_manifest"),
                        "recipe_patch_statuses": list(patch_application.get("patches") or []),
                    }
                )
                live_shared_state.warm_replay_pending = pending
                try:
                    live_shared_state.save(self.session_dir)
                except Exception:
                    log.debug(
                        "combined warm replay pending persist failed",
                        exc_info=True,
                    )
        else:
            applied = patch_application
            pre_patch_sha = before_apply_sha
        return _PatchSettlement(applied=applied, pre_patch_sha=pre_patch_sha)

    @staticmethod
    def _apply_recipe_patches(
        params: Mapping[str, Any],
        output_dir: Any,
        *,
        before_mutation: Any,
    ) -> Any:
        """Apply the recipe's warm patches over the tree they name.

        An overlay recording its own root needs no resolved target, so this only refuses when nothing names a tree
        at all.
        """
        if params.get("patches") and not any(
            str((entry or {}).get("framework_root") or "").strip()
            for entry in params["patches"]
            if isinstance(entry, dict)
        ):
            return {
                "status": "failed",
                "failure": "missing_target_repo",
                "patches": [],
                "applied": [],
            }
        return _apply_warm_patches(
            params,
            "",
            output_dir,
            before_mutation=before_mutation,
        )

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
        headroom_sec, evidence = _round_headroom_sec(state, None)
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
        headroom_sec, evidence = _round_headroom_sec(state, None)
        if headroom_sec is None:
            return True, evidence
        warmup_sec = _positive_seconds(warmup_runtime_sec)
        priced_by = "session_hot_pass"
        benchmark_sec = _phase_state.measured_seconds(state, "baseline_warm_runtime_sec")
        if benchmark_sec is None:
            priced_by = "warmup_post_ready"
            benchmark_sec = _positive_seconds(warmup_post_ready_sec)
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
        pid_dir: Path,
        port: int,
        run_eval: bool | None = None,
    ) -> Path:
        """Render a per-round YAML injecting ``benchmark.server_lifecycle``."""
        with Path(base_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.setdefault("benchmark", {})
        _lifecycle.inject_lifecycle(
            bench,
            cleanup=cleanup,
            pid_dir=pid_dir,
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
        base = Path(pid_dir)
        tag = f"{framework}_{port}"
        for p in (base / f"{tag}.pid", base / f"{tag}.json"):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    async def _run_reported_round(
        self,
        *,
        label: str,
        config_path: Path,
        output_dir: Path,
        recorder: Any = None,
        run_index: int = 0,
        **common: Any,
    ) -> dict[str, Any]:
        """Announce a benchmark round before it blocks, run it, and record it."""
        await report_progress(unit="baseline_round", label=label, status="started")
        started_at = now_iso("seconds")
        started_monotonic = time.monotonic()

        def _record(result: dict[str, Any]) -> None:
            if recorder is None:
                return
            recorder.record_round(
                run_index=run_index,
                label=label,
                started_at=started_at,
                duration_sec=round(time.monotonic() - started_monotonic, 3),
                timeout_sec=common.get("timeout_sec"),
                result=result,
            )

        try:
            result = await self._run_single_benchmark(
                config_path=config_path,
                output_dir=output_dir,
                **common,
            )
        except BaseException as exc:
            # A round that raised still happened, and its row is the only place the timeline can say which round the
            # action died in.
            _record({"status": "failed", "error_class": type(exc).__name__, "error": repr(exc)})
            raise
        _record(result)
        return result

    async def _mn_warmup_pass(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
        output_dir: Path,
        framework: str,
        timeout_sec: int,
        session_deadline_sec: float | None,
        silence_timeout_sec: float | None = None,
        capture_meta: dict[str, Any],
        round_warnings: list[str],
        ctx_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run the discarded multi-node client warmup against the restarted server."""
        # A multi-node round is two client passes of the same shape against a server the round did not boot, and the
        # session's measured figure is one of them -- the round's wall-clock is taken after this pass -- so the pair
        # costs twice it.
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
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
            refused = _stopped_round_result(
                STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
                round_label="multi-node round",
                returncode=None,
                runtime_sec=0.0,
                output_dir=output_dir,
                capture_meta=capture_meta,
                started=False,
            )
            refused["budget_shortfall"] = evidence
            return refused
        warm_dir = output_dir / "mn_warmup"
        started_unix = time.time()
        # The measurement is discarded, but the returncode is not: this pass is a full benchmark round, so a stop here
        # ends the baseline round.
        warm_rc: int | None = None
        try:
            warm_dir.mkdir(parents=True, exist_ok=True)
            warm_cmd = [str(warm_dir) if c == str(output_dir) else c for c in cmd]
            warm_env = dict(env)
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
                    timeout=timeout_sec,
                    silence_timeout_sec=silence_timeout_sec,
                    server_already_ready=True,
                    server_log_path=_watchdog_server_log_path(warm_dir, framework),
                    on_output=warm_activity.note,
                    session_deadline_sec=session_deadline_sec,
                )
            warm_rc = warm_proc.returncode
            log.info("baseline_executor: MN warmup pass done (discarded) rc=%s", warm_rc)
        except subprocess.TimeoutExpired as exc:
            log.warning("baseline_executor: MN warmup pass hit its own hang backstop (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        except Exception as exc:  # noqa: BLE001 - a warmup that fails on its own is best-effort
            log.warning("baseline_executor: MN warmup pass failed (ignored): %r", exc)
            round_warnings.append(_MN_WARMUP_DID_NOT_WARM_WARNING)
        warm_stopped = stopped_by_the_run(warm_rc)
        if warm_stopped is not None:
            return _stopped_round_result(
                warm_stopped,
                round_label="multi-node warmup pass",
                returncode=warm_rc,
                runtime_sec=max(0.0, time.time() - started_unix),
                output_dir=output_dir,
                capture_meta=capture_meta,
            )
        return None

    def _preflight_server_argv(
        self,
        *,
        config_path: Path,
        framework: str,
        launch_env: dict[str, str],
        output_dir: Path,
        attempt: int,
        capture_meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Ask the installed framework's parser about this round's argv.

        Args:
            config_path: The materialised YAML this round launches.
            framework: Framework the config serves.
            launch_env: The environment the benchmark subprocess will get.
            output_dir: The round slot, for the observation artifact.
            attempt: This attempt's index within the slot.
            capture_meta: Result fields echoed onto whatever this round returns.

        Returns:
            dict | None: A terminal failure result when the argv is refused;
            ``None`` when the round may proceed -- including every unavailable
            verdict, which must never cost a round.
        """
        from .magpie_agentx import native_config

        if native_config(config_path):
            return None
        from ...bringup import (
            ARGV_INVALID,
            argv_invalid_observation,
            check_server_argv,
            write_boot_observation,
        )
        from ...bringup.argv_preflight import OK, PARSED_AFTER_DROP, UNAVAILABLE
        from ._server_argv import config_launch_env, config_server_argv, reseal_config_argv

        sealed = config_server_argv(config_path)
        if not sealed.tokenized or not sealed.argv:
            return None

        effective_launch_env = config_launch_env(config_path, launch_env)
        probe_argv = sealed.argv
        if framework.strip().lower() == "vllm" and effective_launch_env.get("PROFILE") == "1":
            probe_argv = (
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                str(output_dir / "torch_trace"),
                *probe_argv,
            )

        # ``_resolve_shared_state`` is typed loosely and callers inject partial
        # doubles, so the round's repair ledger may not be present at all.
        enablement = getattr(self._resolve_shared_state(), "enablement", None)
        spent: list[str] = enablement.argv_repairs if enablement is not None else []
        verdict = check_server_argv(
            framework=framework,
            argv=probe_argv,
            text=sealed.text,
            launch_env=effective_launch_env,
            repaired=spent,
            digest=sealed.digest,
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
                reseal_config_argv(config_path, verdict.text)
                if verdict.repaired_digest:
                    spent.append(verdict.repaired_digest)
                capture_meta["server_argv_dropped"] = list(verdict.dropped)
            return None

        observation = argv_invalid_observation(verdict, session_dir=self.session_dir)
        capture_meta["boot_observation_path"] = write_boot_observation(
            observation,
            session_dir=self.session_dir,
            output_dir=output_dir,
            attempt=attempt,
        )
        excerpt = observation.excerpt
        log.error(
            "argv preflight: %s refused the server argv (%s); not launching",
            framework or "the framework",
            verdict.reason,
        )
        return {
            "status": "failed",
            "error_class": ARGV_INVALID,
            "error": f"{framework or 'framework'} rejected the server argv ({verdict.reason}): {verdict.detail}",
            "output_dir": str(output_dir),
            "enablement_launch_log": excerpt.text if excerpt is not None else "",
            "server_argv": list(verdict.argv),
            **capture_meta,
        }

    async def _restart_multi_node_server(
        self,
        round_: _BenchmarkRound,
        *,
        server_args: str,
        resolved_model: str,
    ) -> dict[str, Any] | None:
        """Give a multi-node round a fresh sglang/vllm server, reporting a restart that failed."""
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        if round_.ctx_extra.get("mn_round_restarted"):
            return None
        try:
            # Merge the reference base UNDER the per-task args (last-wins) so a multi-node per-round restart
            # carries the same reference flags the single-node materialized YAML does.
            from ._grid_runner import merge_server_args
            from ._workload_envs import resolve_reference_launch

            ref_args, ref_envs, _ = resolve_reference_launch()
            # Fold in the operator ``--server-args`` (``INFERENCE_OPTIMIZER_SERVER_ARGS``).
            operator_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
            base = ref_args
            if operator_args:
                base = merge_server_args(base, operator_args) if base else operator_args
            # Base on ``server_args`` (carries the one-shot cuda-graph eager-fallback flag when armed) so the MN
            # per-round restart keeps that fallback too.
            env = {str(k): str(v) for k, v in ref_envs.items()}
            # PD knobs auto-resolved by the helper from $PD_* env, falling back to state.json.
            await restart_server_for_round(
                extra_server_args=merge_server_args(base, server_args) if base else server_args,
                extra_env=env or None,
                framework=os.environ.get("FRAMEWORK") or None,
                model_path=resolved_model or None,
                tp=int(os.environ.get("TP") or 0) or None,
                ep=int(os.environ.get("EP") or 0) or None,
            )
        except ServerRestartFailed as exc:
            return {
                "status": "failed",
                "error_class": "mn_server_restart_failed",
                "error": str(exc),
                "output_dir": str(round_.output_dir),
            }
        return None

    async def _multi_node_warmup_result(
        self,
        round_: _BenchmarkRound,
        round_warnings: list[str],
    ) -> dict[str, Any] | None:
        """Run the multi-node warmup pass, reporting a result only when it ends the round."""
        from ._multi_node_env import (
            is_multi_node,
            mn_bench_warmup_enabled,
        )

        if not (is_multi_node() and mn_bench_warmup_enabled() and not round_.ctx_extra.get("mn_round_restarted")):
            return None
        return await self._mn_warmup_pass(
            cmd=round_.cmd,
            env=round_.env,
            output_dir=round_.output_dir,
            framework=round_.framework,
            timeout_sec=round_.timeout_sec,
            session_deadline_sec=round_.session_deadline_sec,
            silence_timeout_sec=round_.silence_timeout_sec,
            capture_meta=round_.capture_meta,
            round_warnings=round_warnings,
            ctx_extra=round_.ctx_extra,
        )

    @staticmethod
    def _server_is_already_ready(round_: _BenchmarkRound) -> bool:
        """Whether this round attaches to a server that is already up rather than booting one."""
        from ._multi_node_env import is_multi_node

        return bool(round_.server_already_ready or round_.ctx_extra.get("server_already_ready") or is_multi_node())

    def _record_bringup(
        self,
        round_: _BenchmarkRound,
        attempt_index: int,
        workspaces_before: Any,
        *,
        wrapper_stderr: str = "",
        wrapper_stdout: str = "",
    ) -> str:
        """Observe this attempt's boot once and name it on every later result.

        Args:
            round_: The round whose boot is being classified.
            attempt_index: This attempt's index under the round's output dir.
            workspaces_before: The workspaces that existed before the subprocess started.
            wrapper_stderr: The launcher's stderr, fallback evidence.
            wrapper_stdout: The launcher's stdout, fallback evidence.

        Returns:
            str: The server log text that was classified.
        """
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

    async def _sweep_stale_aiter_locks(self, round_: _BenchmarkRound) -> None:
        """Reap aiter JIT locks orphaned by a dead process before this round boots a server over them."""
        if not self.benchmark_watchdog or self._server_is_already_ready(round_):
            return
        from ._aiter_jit import sweep_stale_aiter_locks_if_dead

        lock_sweep = await asyncio.to_thread(sweep_stale_aiter_locks_if_dead)
        if lock_sweep.get("deleted"):
            log.warning(
                "baseline_executor: reaped %d orphaned aiter JIT lock(s) before server launch",
                lock_sweep["deleted"],
            )

    async def _launch_benchmark(self, round_: _BenchmarkRound, started_unix: float) -> _BenchmarkOutput:
        """Run Magpie to completion.

        Launched via ``run_with_session_kill`` so the whole descendant tree is torn down on every exit path
        (plain subprocess.run leaks daemonized server processes).
        """
        server_ready = self._server_is_already_ready(round_)
        if round_.serving_lease is not None:
            # Ray-managed GPU execution (§12 T1): run inside the lease's actor (holds num_gpus across this run's
            # rounds).
            from ._ray_backend import strip_visible_devices_from_config

            ray_cmd = build_benchmark_command(
                python_exe=self.magpie_python,
                config_path=strip_visible_devices_from_config(round_.config_path),
                output_dir=round_.output_dir,
            )
            # No liveness callback is possible here: the round runs inside a Ray actor in another process
            # (potentially on another node) and only its final ``(rc, stdout, stderr)`` crosses back, so there is
            # nothing local to call per line of child output.
            returncode, stdout, stderr = await asyncio.to_thread(
                round_.serving_lease.run_session_kill,
                ray_cmd,
                env=round_.env,
                cwd=str(round_.output_dir),
                timeout=round_.timeout_sec,
                silence_timeout_sec=round_.silence_timeout_sec,
                server_already_ready=server_ready,
                server_log_path=round_.watchdog_server_log,
                session_remaining_sec=session_deadline_to_remaining_sec(round_.session_deadline_sec),
            )
            return _BenchmarkOutput(returncode, stdout, stderr, max(0.0, time.time() - started_unix))
        async with heartbeat_while_output_flows(
            unit="baseline_round",
            label="benchmark",
        ) as activity:
            proc = await asyncio.to_thread(
                run_with_session_kill,
                round_.cmd,
                env=round_.env,
                cwd=str(round_.output_dir),
                timeout=round_.timeout_sec,
                silence_timeout_sec=round_.silence_timeout_sec,
                server_already_ready=server_ready,
                server_log_path=round_.watchdog_server_log,
                on_output=activity.note,
                session_deadline_sec=round_.session_deadline_sec,
            )
        return _BenchmarkOutput(
            proc.returncode,
            proc.stdout,
            proc.stderr,
            max(
                0.0,
                time.time() - started_unix,
            ),
        )

    @staticmethod
    def _timeout_failure(
        round_: _BenchmarkRound,
        started_unix: float,
        workspaces_before: Any,
        exc: subprocess.TimeoutExpired,
    ) -> dict[str, Any]:
        """Report a round the timeout reaped, harvesting whatever it leaked on the way out."""
        destination = select_run_workspace(round_.output_dir, known_before=workspaces_before) or round_.output_dir
        harvested = harvest_leaked_artifacts(
            destination,
            subprocess_started_unix=started_unix,
        )
        return {
            "status": "failed",
            "error_class": "timeout",
            "error": f"baseline benchmark exceeded {round_.timeout_sec}s: {exc}",
            "output_dir": str(round_.output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in harvested],
            **round_.capture_meta,
        }

    @staticmethod
    def _detokenizer_stall_failure(
        round_: _BenchmarkRound,
        output: _BenchmarkOutput,
        started_unix: float,
        workspaces_before: Any,
    ) -> dict[str, Any]:
        """Report a round the detokenizer-stall watchdog reaped: the server came up healthy, then went silent."""
        destination = select_run_workspace(round_.output_dir, known_before=workspaces_before) or round_.output_dir
        harvested = harvest_leaked_artifacts(
            destination,
            subprocess_started_unix=started_unix,
        )
        log.warning(
            "baseline_executor: detokenizer-stall watchdog reaped run "
            "(server ready but log went silent); error_class=detokenizer_stall."
        )
        return {
            "status": "failed",
            "error_class": "detokenizer_stall",
            "returncode": output.returncode,
            "error": (
                "server reported ready but emitted no log output (hung "
                "engine / detokenizer stall); reaped by the "
                "detokenizer-stall watchdog. See server.log."
            ),
            "subprocess_runtime_sec": round(output.runtime_sec, 2),
            "output_dir": str(round_.output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in harvested],
            **round_.capture_meta,
        }

    @staticmethod
    def _benchmark_outcome(
        round_: _BenchmarkRound,
        output: _BenchmarkOutput,
        started_unix: float,
        workspaces_before: Any,
        server_log_text: str,
    ) -> _BenchmarkOutcome:
        """Read the server log back for the classifications every failure branch below is decided on.

        When the server's engine/worker bootstrap dies, the root cause is in server.log, not Magpie's
        stdout/stderr tail, and the liveness watchdog may have reaped the hung parent with
        ``SERVER_DEAD_RETURNCODE``. Detected once here and reused so the failure is classified
        ``server_init_dead``. Backend-agnostic (vLLM + SGLang). The same read answers whether the wall is the
        recoverable cuda-graph capture one (OOM-rooted ones excluded) that arms the one-shot eager retry.
        """
        death_excerpt = server_log_death_excerpt(str(round_.output_dir / "server.log"))
        stderr, stdout = output.stderr or "", output.stdout or ""
        return _BenchmarkOutcome(
            None,
            output.returncode,
            output.stdout,
            output.stderr,
            output.runtime_sec,
            started_unix,
            workspaces_before,
            round_.capture_meta["boot_observation_path"],
            death_excerpt is not None or output.returncode == SERVER_DEAD_RETURNCODE,
            death_excerpt or ("server engine/worker init failed (reaped by liveness watchdog); see server.log"),
            is_aiter_jit_registry_mismatch(server_log_text, stderr, stdout),
            _is_cuda_graph_capture_failure(server_log_text, stderr, stdout),
        )

    async def _observed_benchmark_outcome(self, round_: _BenchmarkRound) -> _BenchmarkOutcome:
        """Launch the benchmark and classify its exit, or report whatever refused it a measurement."""
        workspaces_before = snapshot_workspaces(round_.output_dir)
        started_unix = time.time()
        # Anchor the Magpie parent process cwd to the per-task output_dir.
        round_.output_dir.mkdir(parents=True, exist_ok=True)
        # A reused output_dir may still hold a previous attempt's server.log,
        # whose terminal init markers would be read as this attempt's, and the
        # ready stamp beside it would make this attempt's boot look like it
        # never happened.
        attempt_index = open_bringup_attempt(round_.output_dir)
        clear_server_ready_stamp(str(round_.output_dir / "server.log"))
        refusal = self._preflight_server_argv(
            config_path=round_.config_path,
            framework=round_.framework,
            launch_env=round_.env,
            output_dir=round_.output_dir,
            attempt=attempt_index,
            capture_meta=round_.capture_meta,
        )
        if refusal is not None:
            return _refused_benchmark(refusal)
        await self._sweep_stale_aiter_locks(round_)
        try:
            output = await self._launch_benchmark(round_, started_unix)
        except subprocess.TimeoutExpired as exc:
            # A reaped timeout carries no wrapper streams; the server log is
            # the whole of the evidence.
            self._record_bringup(round_, attempt_index, workspaces_before)
            return _refused_benchmark(self._timeout_failure(round_, started_unix, workspaces_before, exc))
        server_log_text = self._record_bringup(
            round_,
            attempt_index,
            workspaces_before,
            wrapper_stderr=output.stderr or "",
            wrapper_stdout=output.stdout or "",
        )
        stopped = stopped_by_the_run(output.returncode)
        if stopped is not None:
            return _refused_benchmark(
                _stopped_round_result(
                    stopped,
                    round_label="measured round",
                    returncode=output.returncode,
                    runtime_sec=output.runtime_sec,
                    output_dir=round_.output_dir,
                    capture_meta=round_.capture_meta,
                )
            )
        # Detokenizer-stall watchdog reap: the server came up healthy but went silent for the stall grace window
        # (hung engine / wedged detokenizer).
        if output.returncode == DETOKENIZER_STALL_RETURNCODE:
            return _refused_benchmark(self._detokenizer_stall_failure(round_, output, started_unix, workspaces_before))
        return self._benchmark_outcome(round_, output, started_unix, workspaces_before, server_log_text)

    @staticmethod
    def _failed_measurement_result(
        round_: _BenchmarkRound,
        outcome: _BenchmarkOutcome,
        measurement: dict[str, Any],
        warnings: list[str],
        workspace: Path,
    ) -> dict[str, Any]:
        """Name why this round produced no usable measurement and report it as the round's failure."""
        report_path = workspace / _BENCHMARK_REPORT_NAME
        tail = (outcome.stderr or outcome.stdout or "")[-2000:]
        death_or_tail = outcome.server_init_dead_error if outcome.server_init_dead else tail
        if outcome.registry_mismatch:
            error_class, error = "aiter_jit_registry_mismatch", death_or_tail
        elif outcome.cuda_graph_capture_failed:
            error_class, error = "cuda_graph_capture_failed", death_or_tail
        elif outcome.server_init_dead:
            error_class, error = "server_init_dead", outcome.server_init_dead_error
        elif outcome.returncode != 0:
            error_class = _classify_subprocess_error(
                outcome.runtime_sec,
                tail,
                outcome.returncode,
            )
            error = tail
        elif not report_path.exists():
            error_class = "no_report"
            error = f"{_BENCHMARK_REPORT_NAME} missing under {workspace}"
        else:
            error_class = "invalid_measurement"
            error = "benchmark report did not contain positive throughput and completed requests"
        return {
            "status": "failed",
            "error_class": error_class,
            "returncode": outcome.returncode,
            "error": redact_secret_values(error),
            "output_signature": nonzero_output_signature(error_class, outcome.stdout, outcome.stderr),
            "output_dir": str(round_.output_dir),
            "workspace": str(workspace),
            "report_path": str(report_path) if report_path.exists() else None,
            "reported_success": measurement.get("reported_success"),
            "subprocess_runtime_sec": round(outcome.runtime_sec, 2),
            "nonfatal_warnings": warnings,
            **round_.capture_meta,
        }

    @staticmethod
    def _refuse_nonzero_exit_without_complete_protocol(
        round_: _BenchmarkRound,
        outcome: _BenchmarkOutcome,
        measurement: dict[str, Any],
        warnings: list[str],
        workspace: Path,
    ) -> dict[str, Any] | None:
        """Refuse a measurement whose process exited nonzero before it served the protocol it recorded."""
        if outcome.returncode == 0:
            return None
        report_path = workspace / _BENCHMARK_REPORT_NAME
        nonzero_error = redact_secret_values((outcome.stderr or outcome.stdout or "")[-2000:])
        if not served_complete_protocol(measurement):
            return {
                "status": "failed",
                "error_class": "magpie_nonzero_after_valid_measurement",
                "returncode": outcome.returncode,
                "error": nonzero_error,
                "output_dir": str(round_.output_dir),
                "workspace": str(workspace),
                "report_path": str(report_path) if report_path.exists() else None,
                "reported_success": measurement.get("reported_success"),
                "subprocess_runtime_sec": round(outcome.runtime_sec, 2),
                "nonfatal_warnings": warnings,
                **round_.capture_meta,
            }
        # The round served every request the client recorded as requested,
        # so the exit code is not evidence against the measurement. The
        # cause stays on the result rather than only in the log.
        warnings.append(f"nonzero_rc_after_complete_protocol:{outcome.returncode}")
        log.warning(
            "baseline_executor: magpie exited %d after serving its whole protocol (%s of %s requests); "
            "keeping the measurement: %s",
            outcome.returncode,
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
        """Read this round's accuracy eval back onto its result, and report what the round measured."""
        from hyperloom.inference_optimizer import framework_registry

        # Parse accuracy eval results (GSM8K for serving, or the image-quality gate for scriptable frameworks).
        eval_framework = (report or {}).get("framework") or os.environ.get("FRAMEWORK") or None
        # RUN_EVAL gates ONLY the serving lm-eval GSM8K run.
        eval_scriptable = framework_registry.is_scriptable(eval_framework)
        shared_state = (getattr(round_.ctx, "extra", None) or {}).get("shared_state") or self.shared_state
        benchmark_mode = str(getattr(shared_state, "benchmark_mode", "") or "")
        if round_.run_eval_disabled and not eval_scriptable and not is_agentx_mode(benchmark_mode):
            # Serving RUN_EVAL was off this run (a profile round or a staged warmup round), so lm-eval did not
            # execute and there is no fresh accuracy to read.
            log.info(
                "baseline_executor: RUN_EVAL disabled this run (serving); skipping accuracy parse (no lm-eval executed)"
            )
        else:
            self._attach_round_accuracy(result, round_.result_dir, eval_framework, benchmark_mode)
        log.info(
            "baseline_executor: %s %.1f %s (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            result["output_throughput"],
            framework_registry.throughput_unit(eval_framework),
            result["e2el_mean_ms"] or 0.0,
        )

    @staticmethod
    def _attach_round_accuracy(
        result: dict[str, Any],
        result_dir: Path,
        eval_framework: Any,
        benchmark_mode: str,
    ) -> None:
        """Read the accuracy eval this round wrote and carry its verdict onto the round's result."""
        from ._accuracy_gate import eval_probe_summary, parse_eval_results, read_eval_probe

        # Search from ``$RESULT_DIR`` so serving runs survive benchmark_lib.sh moving/cleaning
        # ``$EVAL_RESULT_DIR`` and scriptable quality gates still resolve from Magpie's benchmark reports.
        eval_data = parse_eval_results(
            result_dir,
            framework=eval_framework,
            benchmark_mode=benchmark_mode,
        )
        if eval_data.get("accuracy") is not None:
            result["accuracy"] = eval_data["accuracy"]
            result["accuracy_task"] = eval_data.get("task", "gsm8k")
            result["accuracy_metric"] = eval_data.get("metric", "")
            result["accuracy_source"] = eval_data.get("source_file", "")
            log.info("baseline_executor: accuracy=%.4f (%s)", result["accuracy"], result["accuracy_task"])
        else:
            log.warning("baseline_executor: accuracy eval not found: %s", eval_data.get("error", "unknown"))
        # Records why the score is ~0; the score itself is already correct.
        eval_probe = read_eval_probe(result_dir)
        if eval_probe:
            result["eval_probe"] = eval_probe
            log.warning("baseline_executor: %s", eval_probe_summary(eval_probe))

    async def _round_result_from_workspace(
        self,
        round_: _BenchmarkRound,
        outcome: _BenchmarkOutcome,
        round_warnings: list[str],
    ) -> dict[str, Any]:
        """Turn the workspace the benchmark left behind into this round's result."""
        workspace = select_run_workspace(round_.output_dir, known_before=outcome.workspaces_before)
        # Always-on artifact harvest: copy wrapper-side leaks into the task workspace so failure-path diagnostics
        # survive; mtime gating rejects stale prior-run leaks.
        harvested = harvest_leaked_artifacts(
            workspace if workspace is not None else round_.output_dir,
            subprocess_started_unix=outcome.started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if workspace is None:
            return self._no_workspace_failure(
                _BenchProcess(
                    returncode=outcome.returncode,
                    stdout=outcome.stdout,
                    stderr=outcome.stderr,
                    runtime_sec=outcome.runtime_sec,
                    server_init_dead=outcome.server_init_dead,
                    server_init_dead_error=outcome.server_init_dead_error,
                    registry_mismatch=outcome.registry_mismatch,
                    cuda_graph_capture_failed=outcome.cuda_graph_capture_failed,
                    capture_meta=round_.capture_meta,
                ),
                output_dir=round_.output_dir,
                harvested=harvested,
            )
        report_path = workspace / _BENCHMARK_REPORT_NAME
        report = _loaded_benchmark_report(report_path)
        measurement = extract_benchmark_measurement(
            report,
            config_path=round_.config_path,
            workspace=workspace,
            subprocess_started_unix=outcome.started_unix,
        )
        warnings = round_warnings + list(measurement.pop("nonfatal_warnings", []) or [])
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")
        if not self._measurement_succeeded(measurement):
            return self._failed_measurement_result(round_, outcome, measurement, warnings, workspace)
        refused = self._refuse_nonzero_exit_without_complete_protocol(round_, outcome, measurement, warnings, workspace)
        if refused is not None:
            return refused
        result = self._succeeded_round_result(round_, outcome, measurement, warnings, workspace)
        await asyncio.to_thread(
            _attach_baseline_launch_evidence,
            result,
            config_path=round_.materialized_config_path,
            output_dir=round_.output_dir,
            framework=round_.framework,
        )
        self._stamp_round_quality(round_, result, report, warnings)
        return result

    @staticmethod
    def _succeeded_round_result(
        round_: _BenchmarkRound,
        outcome: _BenchmarkOutcome,
        measurement: dict[str, Any],
        warnings: list[str],
        workspace: Path,
    ) -> dict[str, Any]:
        """The round's measurement, with where every artifact it produced landed."""
        report_path = workspace / _BENCHMARK_REPORT_NAME
        return {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": outcome.returncode,
            "output_dir": str(round_.output_dir),
            "result_dir": str(round_.result_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Materialized YAML for THIS baseline.
            "materialized_config": str(round_.materialized_config_path),
            # Magpie subprocess wall-clock (success path only).
            "subprocess_runtime_sec": round(outcome.runtime_sec, 2),
            # The benchmark's own share of that wall-clock, boot excluded.
            "post_ready_runtime_sec": _round_post_ready_sec(
                round_.watchdog_server_log,
                started_unix=outcome.started_unix,
                runtime_sec=outcome.runtime_sec,
            ),
            # Authoritative (materialized-config) view of whether the serving lm-eval ran this run.
            "run_eval_disabled": bool(round_.run_eval_disabled),
            # Where this round's boot observation landed, and whether the log
            # it was classified from could be read.
            "boot_observation_path": outcome.boot_observation_ref,
            "boot_observation_degraded": round_.capture_meta.get("boot_observation_degraded", ""),
        }

    async def _measured_round_result(self, round_: _BenchmarkRound) -> dict[str, Any]:
        """Run the benchmark subprocess and turn what it left behind into this round's result."""
        from ._multi_node_env import log_mn_banner

        log_mn_banner("baseline_executor", log, output_dir=str(round_.output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s", round_.cmd, round_.output_dir)
        round_warnings: list[str] = []
        warmup_result = await self._multi_node_warmup_result(round_, round_warnings)
        if warmup_result is not None:
            return warmup_result
        outcome = await self._observed_benchmark_outcome(round_)
        if outcome.failure is not None:
            return outcome.failure
        return await self._round_result_from_workspace(round_, outcome, round_warnings)

    async def _run_single_benchmark(
        self,
        *,
        config_path: Path,
        output_dir: Path,
        timeout_sec: int,
        override_result_dir: str | None,
        resolved_model: str,
        materialized_config_path: Path,
        inferencex_path: str,
        effective_extra_server_args: str,
        params: dict[str, Any],
        ctx: RunnerContext,
        run_eval_disabled: bool = False,
        serving_lease: Any = None,
        server_already_ready: bool = False,
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result."""
        from .magpie_agentx import bind_source_evidence

        await asyncio.to_thread(bind_source_evidence, config_path, params.get("patches") or [])
        cmd = build_benchmark_command(
            python_exe=self.magpie_python,
            config_path=config_path,
            output_dir=output_dir,
        )
        result_dir = _resolve_result_dir(output_dir, override_result_dir)
        env = _benchmark_process_env(config_path, output_dir, result_dir, inferencex_path)
        # Config/eval-contract facts echoed onto every failure result so an eval-rooted failure can be fingerprinted
        # and re-run by enablement.
        capture_meta = {
            "materialized_config": str(materialized_config_path),
            "result_dir": str(result_dir),
            "run_eval_disabled": bool(run_eval_disabled),
        }
        # The materialized YAML is authoritative for the framework: params and $FRAMEWORK are both optional on a
        # baseline task.
        framework = (
            _config_framework(materialized_config_path)
            or str(params.get("framework") or "").strip().lower()
            or os.environ.get("FRAMEWORK", "").strip().lower()
        )
        # Multi-node (--nodes >= 2): inject MAGPIE_RUN_PHASE=client + BENCHMARK_BASE_URL so Magpie skips its server
        # launch and targets the RayJob head.
        from ._multi_node_env import magpie_remote_env

        env.update(magpie_remote_env())
        ctx_extra = getattr(ctx, "extra", None) or {}
        # The session's wall-clock budget, resolved per round rather than once per task: a baseline runs up to three
        # of them, and a warmup that overran has already spent budget the ones after it were counting on.
        session_deadline_sec, _ = session_grid_bounds(ctx_extra.get("shared_state") or self.shared_state)
        silence_timeout_sec = None
        if self.benchmark_watchdog:
            silence_timeout_sec, timeout_sec = resolve_benchmark_timeouts()
            env["PYTHONUNBUFFERED"] = "1"
            sync_benchmark_timeout(config_path, timeout_sec)
        round_ = _BenchmarkRound(
            config_path=config_path,
            materialized_config_path=materialized_config_path,
            output_dir=output_dir,
            result_dir=result_dir,
            ctx=ctx,
            ctx_extra=ctx_extra,
            cmd=cmd,
            env=env,
            framework=framework,
            watchdog_server_log=_watchdog_server_log_path(output_dir, framework),
            capture_meta=capture_meta,
            timeout_sec=timeout_sec,
            silence_timeout_sec=silence_timeout_sec,
            session_deadline_sec=session_deadline_sec,
            serving_lease=serving_lease,
            server_already_ready=server_already_ready,
            run_eval_disabled=run_eval_disabled,
        )
        # Multi-node only: restart sglang/vllm per round for a fresh server.
        restart_failure = await self._restart_multi_node_server(
            round_,
            server_args=effective_extra_server_args,
            resolved_model=resolved_model,
        )
        if restart_failure is not None:
            return restart_failure
        return await self._measured_round_result(round_)

    def _no_workspace_failure(
        self,
        process: _BenchProcess,
        *,
        output_dir: Any,
        harvested: Any,
    ) -> dict[str, Any]:
        """Report a run that finished without ever creating a benchmark workspace.

        Magpie never created a ``benchmark_*`` directory, so the wrapper never
        wrote a server log: whatever the subprocess printed is the only evidence
        there is, and the error class names which of the known causes it was.
        """
        failure_extras = {
            "output_dir": str(output_dir),
            "harvested_artifacts": [str(dst) for _, dst in harvested],
            **process.capture_meta,
        }
        # Magpie never created a benchmark_* workspace, so the wrapper never wrote server.log.
        captured = redact_secret_values((process.stderr or "") + (process.stdout or ""))
        stderr_log_path: str | None = None
        if captured.strip():
            try:
                log_file = output_dir / "baseline_stderr.log"
                log_file.write_text(captured, encoding="utf-8")
                stderr_log_path = str(log_file)
            except OSError as exc:
                log.warning(
                    "baseline_executor: failed to persist stderr log: %s",
                    exc,
                )
        if stderr_log_path:
            failure_extras["stderr_log_path"] = stderr_log_path
        # Registry mismatch is wrapped in "Capture cuda graph failed" and must win so we do not arm the
        # disable-cuda-graph retry.
        if process.registry_mismatch:
            return {
                "status": "failed",
                "error_class": "aiter_jit_registry_mismatch",
                "returncode": process.returncode,
                "error": redact_secret_values(
                    process.server_init_dead_error
                    if process.server_init_dead
                    else (process.stderr or process.stdout or "")[-2000:]
                ),
                **failure_extras,
            }
        # cuda-graph capture failures take priority over server_init_dead: only this class arms the one-shot
        # disable-cuda-graph retry.
        if process.cuda_graph_capture_failed:
            return {
                "status": "failed",
                "error_class": "cuda_graph_capture_failed",
                "returncode": process.returncode,
                "error": redact_secret_values(
                    process.server_init_dead_error
                    if process.server_init_dead
                    else (process.stderr or process.stdout or "")[-2000:]
                ),
                **failure_extras,
            }
        if process.server_init_dead:
            return {
                "status": "failed",
                "error_class": "server_init_dead",
                "returncode": process.returncode,
                "error": redact_secret_values(process.server_init_dead_error),
                **failure_extras,
            }
        if process.returncode != 0:
            return self._nonzero_exit_failure(process, failure_extras)
        return {
            "status": "failed",
            "error_class": "no_workspace",
            "error": "Magpie completed but produced no benchmark_* workspace",
            **failure_extras,
        }

    @staticmethod
    def _nonzero_exit_failure(process: _BenchProcess, failure_extras: dict[str, Any]) -> dict[str, Any]:
        """Report a benchmark subprocess that exited nonzero, classified by its tail."""
        tail = redact_secret_values((process.stderr or process.stdout or "")[-2000:])
        err_class = _classify_subprocess_error(
            process.runtime_sec,
            tail,
            process.returncode,
        )
        return {
            "status": "failed",
            "error_class": err_class,
            "returncode": process.returncode,
            "subprocess_runtime_sec": round(process.runtime_sec, 2),
            "error": tail,
            "output_signature": nonzero_output_signature(err_class, process.stdout, process.stderr),
            **failure_extras,
        }

    def _roll_back_failed_patches(
        self,
        patch_application: Mapping[str, Any],
        *,
        params: Mapping[str, Any],
        output_dir: Any,
        live_shared_state: Any,
    ) -> _PatchSettlement:
        """Unwind an apply that failed, and report it when the timeline required it.

        The kernel half is rolled back first, then the recipe half, so the tree is
        whole again before anything is reported.
        """
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
        rollback_result = {
            "ok": bool(recipe_rollback.get("ok") and kernel_rollback.get("ok")),
            "recipe": recipe_rollback,
            "kernel": kernel_rollback,
            "errors": rollback_errors,
        }
        if live_shared_state is not None:
            if rollback_result["ok"]:
                live_shared_state.warm_replay_pending = {}
            else:
                live_shared_state.warm_replay_pending = {
                    **dict(
                        getattr(
                            live_shared_state,
                            "warm_replay_pending",
                            {},
                        )
                        or {}
                    ),
                    "status": "rollback_failed",
                    "rollback_errors": rollback_errors,
                }
                if hasattr(live_shared_state, "set_stop_reason"):
                    live_shared_state.set_stop_reason("warm_replay_rollback_failed")
            try:
                live_shared_state.save(self.session_dir)
            except Exception:
                log.warning(
                    "combined warm replay cleanup persist failed",
                    exc_info=True,
                )
        return _PatchSettlement(
            failure={
                "status": ("required_patch_failed" if rollback_result["ok"] else "required_patch_rollback_failed"),
                "error_class": ("required_patch_failed" if rollback_result["ok"] else "warm_replay_rollback_failed"),
                "error": str(patch_application.get("failure") or "required recipe timeline patch failed"),
                "required_patch_failure": patch_application,
                "failed_patch_ref": patch_application.get("failed_ref"),
                "warm_kernel_rolled_back": bool(kernel_rollback.get("ok")),
                "warm_replay_rollback": rollback_result,
                "workspace": str(output_dir),
            }
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
        # Required custom tokenizers must reach the throughput client as well as eval.
        tokenizer_failure = self._client_tokenizer_hook_failure(config_path, ix_root)
        if tokenizer_failure is not None:
            return tokenizer_failure
        if materialized_run_eval_disabled(config_path):
            return None
        return self._eval_patch_failure(ix_root)

    def _client_tokenizer_hook_failure(self, config_path: Path, ix_root: str) -> dict[str, Any] | None:
        """Install the client-side tokenizer hook this model needs, reporting a tree it cannot be installed in."""
        tok_mode = _client_tokenizer_mode(self._model_from_config(config_path))
        if not tok_mode:
            return None
        try:
            hook_ok = ensure_client_tokenizer_hook(
                inferencex_dir=ix_root or None,
                script_name=self._client_script_from_config(config_path),
            )
        except OSError as exc:
            log.error("baseline_executor: client tokenizer hook raised for %s: %s", ix_root, exc)
            hook_ok = False
        if hook_ok:
            return None
        msg = (
            f"the benchmark client cannot be told to load the {tok_mode!r} tokenizer: "
            f"the hook could not be installed in InferenceX's benchmark scripts "
            f"(inferencex={ix_root or '<unset>'}). This model's client resolves the "
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

    def _eval_patch_failure(self, ix_root: str) -> dict[str, Any] | None:
        """Patch the InferenceX tree this run's lm-eval needs, reporting a tree that will abort the eval."""
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
            task_id=str(getattr(ctx.task, "task_id", "") or ""),
            task_kind=str(getattr(ctx.task, "kind", "") or ""),
            reason=str(params.get("reason") or ""),
            framework=self._resolve_framework(ctx),
            establishes_quality_ref=_should_establish_quality_ref(getattr(ctx.task, "kind", ""), params),
            params=params,
            failure_streak_before=streak,
            total_failures_before=total,
            owns_event=not str(params.get(INLINE_EVENT_PARAM) or ""),
        )
        try:
            result = await self._run_retrying(ctx, recorder=recorder)
            shared_state = (getattr(ctx, "extra", None) or {}).get("shared_state")
            if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params) and agentx_active(shared_state):
                refuse_cacheless_agentx_baseline(result)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        if recorder is not None:
            recorder.finish(result)
        return result

    def _resolve_sink(self, ctx: RunnerContext) -> Any:
        """Decide which event this measurement's rows belong to."""
        from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import baseline_event_id
        from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
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
            return make_sink(inline, producer=_RECORDER_PRODUCER)
        state = self._resolve_shared_state((getattr(ctx, "extra", None) or {}).get("shared_state"))
        event = baseline_event_id(
            str(getattr(state, "phase", "") or "unphased"),
            int(getattr(state, "macro_cycle", 0) or 0),
        )
        return make_sink(event, producer=_RECORDER_PRODUCER)

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
    "BenchmarkRunExecutor",
    "baseline_executor",
]
