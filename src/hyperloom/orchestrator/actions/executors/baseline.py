# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

Runs the Magpie CLI as a subprocess, parses ``benchmark_report.json``,
and returns the result on the bus as a ``delegated_result`` event.

RunnerContext.task.params keys (all optional; defaults from
default_baseline_config()): ``config_path``, ``output_dir``, ``timeout_sec``.

Returns ``error_class`` on failure so the coordinator can route to
Robustness RCA later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import redact_secret_values, scrub_benchmark_process_env
from hyperloom.common.git_safety import safe_directory_args
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...framework.paths import resolve_session_framework_root
from ...loop.sub_agent_runner import RunnerContext
from ...trace.task_progress import heartbeat_while_output_flows, report_progress
from ...phases import machine_state as _phase_state
from ..stop_attribution import (
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from . import _server_lifecycle as _lifecycle
from ._file_lock import best_effort_file_lock
from ._aiter_jit import (
    AITER_JIT_PROBE_PATHS,
    BASELINE_COLD_START_TIMEOUT_SEC,
    COLD_START_KERNEL_THRESHOLD,
    probe_aiter_jit_cache as _probe_aiter_jit_cache,
    sweep_stale_aiter_locks_if_dead,
)

# The grid module is the namespace the helpers both benching arms share ended up
# in: how a sentinel returncode reads back, how a round's cap is clamped to the
# budget, how the two session bounds are resolved, and the hygiene every launch
# needs. Imported rather than restated here so a baseline round and a grid round
# are priced the same way. The returncode decoder's class-side sibling is in
# ``..stop_attribution``, which says why the two sides sit where they do.
from ._grid_runner import (
    _kill_stale_servers,
    sanitize_result_dir,
    sanitize_script_name,
    session_clamped_timeout_sec,
    session_grid_bounds,
    stopped_by_the_run,
)
from ._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    clear_server_ready_stamp,
    post_ready_runtime_sec,
    run_with_session_kill,
    server_log_death_excerpt,
    session_deadline_to_remaining_sec,
)
from ._accuracy_gate import (
    _RUN_EVAL_FALSE_VALUES,
    materialized_run_eval_disabled,
)
from ._workload_envs import (
    _remove_moe_runner_backend_arg,
    FrameworkScriptMismatchError,
    agentx_enabled,
    default_baseline_config,
    materialize_config_with_envs,
    prepare_agentx_runtime,
)
from ._inferencex_patcher import (
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_lib_eval_start_patched,
    ensure_eval_probe_patched,
    eval_probe_targets_exist,
    failed_patch_anchors,
)
from ._magpie_patcher import ensure_eval_concurrency_compat
from ._patch_snapshot import (
    _create_patch_snapshot,
    _patch_touched_paths_from_text as _patch_touched_paths,
    _restore_patch_snapshot,
)
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    select_run_workspace,
    snapshot_workspaces,
)
from .benchmark_backend import build_benchmark_command


log = logging.getLogger(__name__)


# Markers identifying an InferenceX ``run_eval`` (lm-eval) failure as the root
# cause of a benchmark non-zero exit.
_EVAL_FAILURE_MARKERS = (
    "run_eval failed with exit code",
    "ERROR: run_eval failed",
    "Unknown parameter: --concurrent-requests",
)
# Bounded per-file read so log scanning never slurps a multi-GB server.log.
_LOG_SCAN_MAX_BYTES = 262_144
# The measured pass ran as the first traffic against a freshly restarted server,
# because the warmup that exists to drive it did not. Its throughput is a cold
# number, and the session anchors every later gain on it.
_MN_WARMUP_DID_NOT_WARM_WARNING = "baseline_mn_warmup_did_not_run"
# The round kept its warmup pass as the baseline because the budget could not
# pay for the measured pass after it. Same consequence as the warning above --
# a cold anchor -- reached from the other direction, and carried on the result
# so a reader of the session's gains knows the denominator is depressed.
MEASURE_ROUND_DROPPED_WARNING = "baseline_measure_round_dropped_low_budget"

# The cold-start guard's two round directories. The warmup round is the only one
# that measures accuracy (``RUN_EVAL=true``); the measured round is hot
# throughput alone, so it carries no accuracy by construction.
_WARMUP_ROUND_DIR = "warmup_round"
_MEASURE_ROUND_DIR = "measure_round"
_DOUBLE_RUN_ROUND_DIRS = (_WARMUP_ROUND_DIR, _MEASURE_ROUND_DIR)

# Markers identifying a MoE quant scheme with no implementation for the
# ``--moe-runner-backend`` in use: ``create_moe_runner`` falls through without
# building a runner and the first forward pass dies (e.g. Quark MXFP4 on
# triton). Both a missing-runner marker AND a MoE-scheme marker must appear, so
# an unrelated AttributeError mentioning some other ``runner`` attribute does
# not burn a full benchmark retry. Deliberately not keyed on Quark: sglang has
# other aiter-only MoE schemes (quark_int4fp8_moe, the mxfp4 dynamic-quant
# method) that fail exactly the same way.
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


# Fast-exit arg errors (vLLM/sglang exits in <30s on bad CLI args)
# should not consume the slow-baseline retry budget.
FAST_EXIT_THRESHOLD_SEC = 30.0
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

# KV-cache OOM: weights loaded but no room left for the KV cache. Kept distinct
# from a generic nonzero exit so it's attributed to an over-aggressive
# --mem-fraction-static. Surfaces after weight load, so matched regardless of
# the fast-exit elapsed threshold.
_KV_CACHE_OOM_MARKERS = (
    "no gpu memory for the kv cache",
    "leave no gpu memory",
    "raise --mem-fraction-static above",
)


# Strong cuda-graph capture markers: stream-capture incompatibility, reliably
# recoverable by disabling cuda-graph.
_CUDA_GRAPH_STRONG_MARKERS = (
    "operation not permitted when stream is capturing",
    "hiperrorstreamcaptureunsupported",
)
# Weak marker: bare "Capture cuda graph failed" carries no root cause. Trusted
# only when the nearby context is neither OOM nor a compile/lowering error.
_CUDA_GRAPH_WEAK_MARKER = "capture cuda graph failed"

# Profile-cuda-graph shape discovery triggers a recoverable AssertionError that
# must win over the generic assertionerror non-recoverable gate below. Both
# markers are required so a generic AssertionError never matches.
_CUDA_GRAPH_PROFILE_ASSERT_MARKERS = (
    "get_num_new_pages",
    "seq_lens.device == cpu_device",
)

# OOM-rooted capture failures are NOT recoverable by disabling cuda-graph
# (eager peaks can be higher); compile/lowering errors are not either.
_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
)
_NON_RECOVERABLE_MARKERS = (
    "loweringexception",
    "assertionerror",
    "compilationerror",
)
# Strong markers are high-confidence, so OOM exclusion is scoped tight (±1
# line): only an OOM on/adjacent to the marker line demotes it. The bare weak
# marker keeps the whole-blob OOM/compile exclusion.
_STRONG_OOM_CONTEXT_RADIUS = 1


def _is_cuda_graph_capture_failure(*texts: str) -> bool:
    """True when a cuda-graph capture marker is recoverable by disabling graph.

    A strong stream-capture marker arms the fallback unless an OOM sits on its
    ±1-line context (tight, since the marker itself is high confidence). The
    bare ``Capture cuda graph failed`` is a weak signal: the WHOLE blob must
    carry neither OOM nor a compile/lowering error, both unrecoverable by
    disabling cuda-graph. Strong wins on a line that also matches weak, so the
    compile/OOM whole-blob gate never demotes a genuine stream-capture failure.

    Args:
        *texts: Log / stdout / stderr blobs to scan for cuda-graph markers.

    Returns:
        ``True`` when a cuda-graph capture failure recoverable by disabling
        cuda-graph capture is detected, else ``False``.
    """
    lines = "\n".join(t for t in texts if t).splitlines()
    lowered = [ln.lower() for ln in lines]
    blob = "\n".join(lowered)
    # Profile-cuda-graph assert wins over the assertionerror gate.
    if all(m in blob for m in _CUDA_GRAPH_PROFILE_ASSERT_MARKERS):
        return True
    blob_has_oom = any(m in blob for m in _OOM_MARKERS)
    blob_has_non_recoverable = any(m in blob for m in _NON_RECOVERABLE_MARKERS)
    saw_pure_weak = False
    for idx, line in enumerate(lowered):
        is_strong = any(m in line for m in _CUDA_GRAPH_STRONG_MARKERS)
        if is_strong:
            lo = max(0, idx - _STRONG_OOM_CONTEXT_RADIUS)
            hi = min(len(lowered), idx + _STRONG_OOM_CONTEXT_RADIUS + 1)
            if not any(m in "\n".join(lowered[lo:hi]) for m in _OOM_MARKERS):
                return True
            continue
        if _CUDA_GRAPH_WEAK_MARKER in line:
            saw_pure_weak = True
    if saw_pure_weak and not blob_has_oom and not blob_has_non_recoverable:
        return True
    return False


# Disable cuda-graph capture per framework: sglang uses --disable-cuda-graph,
# vllm uses --enforce-eager.
_DISABLE_CUDA_GRAPH_FLAGS = {
    "sglang": "--disable-cuda-graph",
    "vllm": "--enforce-eager",
}


def _config_framework(config_path: Path | str) -> str:
    """Framework name from a materialized benchmark YAML (``""`` when unreadable)."""
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return ""
    return str((cfg.get("benchmark") or {}).get("framework") or "").strip().lower()


def _watchdog_server_log_path(output_dir: Path, framework: str) -> str | None:
    """``server.log`` path for the subprocess watchdogs, or None when server-less.

    A scriptable framework never writes a ``server.log``, so passing one would
    arm the dead/stall watchdogs against a file that can never exist.
    """
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
    """How much of a round's wall-clock was the benchmark rather than the boot.

    Splitting the two is what lets later work be priced on what it will actually
    cost: an explore variant boots its own server and pays both, while a pass
    that re-attaches to a server already up pays only the second.

    Args:
        server_log_path: The round's ``server.log``, or ``None`` for a scriptable
            framework, which runs no server and therefore has no split to make.
        started_unix: When the round was spawned.
        runtime_sec: The round's full wall-clock.

    Returns:
        float | None: Seconds after the server reported ready, or ``None`` when
        the round has no such boundary or nothing recorded one.
    """
    if not server_log_path:
        return None
    return post_ready_runtime_sec(
        server_log_path,
        started_unix=started_unix,
        runtime_sec=runtime_sec,
    )


def _logged_session_clamp(timeout_sec: int, clamped: int, *, output_dir: Path) -> int:
    """Announce a round's cap being cut by the session budget, and return the cut.

    Every pass of a baseline round derives its cap from the same budget but on its
    own terms, so the line names the round it belongs to.

    Args:
        timeout_sec: The cap the round would have had on an unbounded budget.
        clamped: The cap the budget leaves it; equal to ``timeout_sec`` when the
            budget was never the binding constraint, which logs nothing.
        output_dir: The round's workspace, whose name identifies the pass.

    Returns:
        int: ``clamped``, unchanged.
    """
    if clamped != timeout_sec:
        log.info(
            "baseline_executor: timeout clamped %ds -> %ds by the session budget (round=%s)",
            timeout_sec,
            clamped,
            output_dir.name,
        )
    return clamped


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
    """Build the result for a round the run itself stopped.

    The session budget elapsed mid-round, or the orchestrator cancelled the
    action. Classified apart from every measurement failure -- and checked before
    them, because the reap leaves exactly the evidence a broken server does (no
    workspace, no report, a non-zero returncode) and being graded as
    ``server_init_dead`` or ``subprocess_nonzero`` would put a verdict on the
    model that this round never reached. Every round the baseline runs goes
    through here, the discarded multi-node warmup pass included: a stop in the
    round that warms the server means the round is over, and going on to the
    measured pass would spend GPU time the run has been told to stop spending.
    Nothing here arms a retry either: the cause is the run, and a resume meets it
    again.

    A round the budget stopped *before* it booted anything is the same cause and
    carries the same class; only the wording and the absent returncode differ, so
    a reader is told whether GPU time was spent.

    Args:
        stopped: How to record the cause.
        round_label: Which round was stopped, for the log line.
        returncode: The stopped round's returncode, or ``None`` when nothing ran.
        runtime_sec: Wall-clock seconds the round had run for.
        output_dir: The task workspace, echoed onto the result.
        capture_meta: Config/eval-contract facts every failure result carries.
        started: Whether the round had begun. ``False`` selects the wording for
            work that never launched.

    Returns:
        dict[str, Any]: The failed result carrying the stop's own error class.
    """
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
    """Seconds this round's budget may still spend, and the numbers behind it.

    The session's own usable remainder, which is what every other admission
    decision reads, so a round cannot be judged against a figure the rest of the
    run disagrees with. Not a share of it: the share held back for the
    optimization phases
    (:func:`~...phases.machine_state.prelude_affordable_seconds`) sizes the
    *optional* arms of preparation, where the question is proportion. A round is
    a feasibility question, and answering it with a percentage refuses rounds
    that fit -- a session with ninety minutes left and a twelve-minute pass to
    run is told it has six.

    Falls back to the session deadline for a caller with no session state at all,
    and to no bound when there is neither.

    Args:
        state: The session ``SharedState``, or ``None`` when the caller has no
            session context (direct executor invocation, tests).
        session_deadline_sec: Monotonic-clock session deadline, or ``None``.

    Returns:
        tuple[float | None, dict[str, Any]]: The headroom, or ``None`` when the
            round is under no budget at all, plus the evidence behind it.
    """
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
    """Keep the warmup's cold figure as the anchor, marked as the cold one.

    Two things end a round once its warmup has already paid for the boot, the
    compile and the capture: the budget cannot cover a second pass, and the
    session's budget reaped the second pass mid-flight. Either way a number
    exists and the GPU time behind it is spent, so it is kept rather than
    discarded -- a marked cold anchor beats no anchor. The marker is what tells a
    reader of the session's later gains that their denominator is depressed.

    Args:
        warmup_result: The succeeded warmup pass's result, mutated in place.
        dropped: Why the measured round did not produce the figure, recorded so
            the decision is legible in the result and the session record.

    Returns:
        dict[str, Any]: ``warmup_result``, marked.
    """
    warnings = warmup_result.setdefault("nonfatal_warnings", [])
    if MEASURE_ROUND_DROPPED_WARNING not in warnings:
        warnings.append(MEASURE_ROUND_DROPPED_WARNING)
    warmup_result["measure_round_dropped"] = dropped
    return warmup_result


def _a_use_must_follow_the_round(state: Any) -> bool:
    """Whether this round is only worth running if something can be measured after it.

    A PRELUDE baseline is not a result. It is the denominator later results are
    read against and the anchor their overtime kill uses, so a session that
    cannot afford one variant after it would spend the wall-clock on a number
    nothing ever reads.

    A re-baseline in a later phase is the opposite: it re-measures the stack the
    session has assembled, and that measurement is the deliverable. Requiring a
    successor would refuse exactly the round that validates the run's own answer,
    at the point in the budget where it is most likely to be the last thing left.

    Args:
        state: The session ``SharedState``, or ``None``.

    Returns:
        bool: ``True`` while the round's worth depends on a successor.
    """
    phase = str(getattr(state, "phase", "") or "").strip().upper()
    return phase == _phase_state.PHASE_PRELUDE


def _positive_seconds(value: Any) -> float | None:
    """Coerce a duration a round reported to seconds, or ``None`` when it did not.

    Absent, unparseable and zero are one answer: nothing was measured. None of
    them may read as work that took no time, which is what a plain
    ``float(... or 0.0)`` would make of them.

    Args:
        value: The reported duration, from a round's result.

    Returns:
        float | None: The seconds, or ``None`` when there is no measurement.
    """
    try:
        seconds = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0.0 else None


def _disable_cuda_graph_flag(framework: str) -> str:
    """Return the framework-correct flag that disables cuda-graph capture.

    Args:
        framework: Framework name (e.g. ``"sglang"`` or ``"vllm"``); matched
            case-insensitively.

    Returns:
        The disable-cuda-graph flag for ``framework``, defaulting to
        ``--disable-cuda-graph`` for unknown frameworks.
    """
    return _DISABLE_CUDA_GRAPH_FLAGS.get(
        (framework or "").strip().lower(),
        "--disable-cuda-graph",
    )


def _with_cuda_graph_disabled(extra_server_args: str, framework: str) -> str:
    """Append the framework-correct disable-cuda-graph flag once (idempotent).

    Token-level dedup so a longer flag (e.g. ``--disable-cuda-graph-extra``)
    is not mistaken for an existing ``--disable-cuda-graph``.

    Args:
        extra_server_args: Existing extra server args string (may be empty).
        framework: Framework name used to pick the correct disable flag.

    Returns:
        ``extra_server_args`` with the framework-correct disable-cuda-graph
        flag appended once; unchanged when the flag is already present.
    """
    flag = _disable_cuda_graph_flag(framework)
    if flag in (extra_server_args or "").split():
        return extra_server_args or ""
    return f"{extra_server_args} {flag}".strip()


def _classify_subprocess_error(
    elapsed_sec: float,
    stderr_tail: str,
) -> str:
    """Return 'fast_exit_arg_error' when the subprocess died fast on an arg
    validation error, else 'subprocess_nonzero'.

    Args:
        elapsed_sec: Subprocess wall-clock runtime in seconds.
        stderr_tail: Tail of the subprocess stderr used for marker matching.

    Returns:
        ``"fast_exit_arg_error"`` for a fast exit caused by argument
        validation, else ``"subprocess_nonzero"``.
    """
    tail = (stderr_tail or "").lower()
    # KV-cache OOM can surface long after weight load; match before the
    # fast-exit elapsed gate below.
    if any(m in tail for m in _KV_CACHE_OOM_MARKERS):
        return "kv_cache_oom"
    if elapsed_sec >= FAST_EXIT_THRESHOLD_SEC:
        return "subprocess_nonzero"
    if any(p.lower() in tail for p in _ARG_ERROR_PATTERNS):
        return "fast_exit_arg_error"
    if "valueerror:" in tail and any(p in tail for p in _ARG_ERROR_CONTEXT_PATTERNS):
        return "fast_exit_arg_error"
    return "subprocess_nonzero"


BASELINE_DEFAULT_TIMEOUT_SEC = 7800  # WARM-start cap, 130 min

# An AgentX baseline does not fit either of the caps above, and the cold-start
# detector cannot see why. That detector counts .so files across the whole aiter
# JIT dir and calls it warm above 20 -- but the signature it is really about is
# (model, dtype, TP, max_model_len), and AgentX is the thing that moves
# max_model_len (6144 -> the model's native window). So the first AgentX round on
# any box that has run synthetic work is detected WARM, handed 7800s, and then
# pays the 30+ minute first-compile for a signature it has never built. Measured
# rounds are 4774s (SGLang) and 6676s (vLLM) before that compile; adding it, plus
# a cold corpus mmap (~840s), lands at ~9316s. Neither cap covers it, and neither
# escape hatch reaches it: the cold-cap env var is only read when the probe says
# cold, and nothing in the tree writes params["timeout_sec"] for a baseline. So a
# first AgentX run does not merely risk the timeout -- it hits it, and a baseline
# timeout kills the session before the search starts.
#
# Derived rather than pinned, because the one part of the round that is chosen
# rather than measured is the measurement window: total = setup + corpus + warmup
# + AGENTX_DURATION + mapping. Deriving keeps the cap correct when an operator
# changes the window, instead of leaving a second number to remember.
AGENTX_BASELINE_OVERHEAD_SEC = 7200  # setup + corpus + warmup + first-compile
AGENTX_DEFAULT_DURATION_SEC = 3600  # mirrors aiperf_client.sh's default

# The warmup share of that overhead is not a constant either, and it is the
# share that actually varies by model: aiperf_client.sh bounds the warmup drain
# with AGENTX_WARMUP_GRACE_PERIOD, so a model whose warmup runs long is a model
# whose operator has already had to raise that knob for the round to complete at
# all. Splitting the flat 7200 at its canonical grace lets the cap follow that
# same knob instead of asking for a second, independent number that means the
# same thing -- which is how the constant came to be wrong for Kimi-K3 (a raw
# aiperf run measured warmup alone at ~12075s, past this entire cap). At
# canonical settings the sum is unchanged, so the measured GLM-5.2/Qwen3.8
# calibration this number carries is preserved exactly.
AGENTX_CANON_WARMUP_GRACE_SEC = 1800  # aiperf_client.sh's CANON_WARMUP_GRACE
_AGENTX_NON_WARMUP_OVERHEAD_SEC = AGENTX_BASELINE_OVERHEAD_SEC - AGENTX_CANON_WARMUP_GRACE_SEC

# ...and the warmup share does not only vary by model, it varies by CONCURRENCY,
# which the grace knob cannot express because it is one flat number. The client
# builds warmup as CANON_WARMUP_PER_LANE requests per lane across CONC lanes, so
# the work is linear in CONC *by construction*; a grace chosen at one
# concurrency is arithmetically wrong at another. Measured on Kimi-K3:
#
#     conc=8   ->  87 warmup requests, ~3000s        (10.9 req/lane)
#     conc=16  -> 177 warmup requests, ~5000s        (11.1 req/lane)
#     conc=64  -> the 12075s warmup the warning below cites
#
# Left flat, a conc=32 round derives its cap from a conc=8 budget and is killed
# mid-warmup -- the exact failure this module's cap-raise exists to prevent, just
# moved one axis over. So the warmup share carries a CONC-scaled FLOOR.
#
# Anchored at 8 rather than at a value invented for the purpose: 8 is the lowest
# concurrency at which this repo has a measured Kimi-K3 agentic warmup, and the
# official ladder starts at 4, so at or below it the derivation is unchanged and
# every previously-validated round keeps its exact cap. The floor only ever
# RAISES the cap. That asymmetry is deliberate: a cap that is too large costs
# nothing but a longer wait on a genuinely hung round (and the session budget
# clamps it anyway via ``session_clamped_timeout_sec``), while a cap that is too
# small kills a round that would have finished.
AGENTX_CANON_WARMUP_CONC = 8


def _agentx_conc(src: "Mapping[str, str]") -> int:
    """Concurrency for the round, from CONC; 0 when unset/unparseable."""
    raw = (src.get("CONC") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0


def agentx_baseline_timeout_sec(env: "Mapping[str, str] | None" = None) -> int:
    """Resolve the AgentX baseline cap: explicit, else duration + overhead.

    Args:
        env: Environment to read; defaults to the process environment.

    Returns:
        The subprocess timeout in seconds for an AgentX baseline launch.
    """
    src = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        raw = (src.get(name) or "").strip()
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    def _is_valid_override(name: str) -> bool:
        raw = (src.get(name) or "").strip()
        try:
            return int(raw) > 0
        except ValueError:
            return False

    explicit = _int("AGENTX_BASELINE_TIMEOUT_SEC", 0)
    if explicit:
        return explicit

    # Same validity bar as `_int` itself (parses to a positive int) rather than
    # "non-empty string" -- otherwise an invalid override (e.g. "abc" or "-1")
    # both silently falls back to the default AND suppresses the warning meant
    # to flag exactly that case.
    if _is_valid_override("AGENTX_BASELINE_OVERHEAD_SEC"):
        overhead = _int("AGENTX_BASELINE_OVERHEAD_SEC", AGENTX_BASELINE_OVERHEAD_SEC)
        grace = None
    else:
        # Derive the warmup share from the same knob that bounds it in the
        # client, so the cap tracks the round the operator actually configured
        # rather than the one this constant was measured on.
        grace = _int("AGENTX_WARMUP_GRACE_PERIOD", AGENTX_CANON_WARMUP_GRACE_SEC)
        # CONC-scaled floor: the client's warmup is linear in CONC by
        # construction (per-lane requests x CONC lanes), but the grace knob is a
        # flat number, so a grace chosen at one concurrency under-budgets every
        # higher one. Scale, never shrink, and stay identity at or below the
        # anchor so previously-validated rounds keep their exact cap.
        conc = _agentx_conc(src)
        if conc > AGENTX_CANON_WARMUP_CONC:
            scaled = (grace * conc) // AGENTX_CANON_WARMUP_CONC
            log.info(
                "agentx_baseline_timeout_sec: scaling the warmup share %ds -> %ds for "
                "CONC=%d (warmup work is linear in CONC; anchor CONC=%d). The floor only "
                "raises the cap -- an over-large cap costs a longer wait on a hung round, "
                "an under-sized one kills a round that would have finished.",
                grace,
                scaled,
                conc,
                AGENTX_CANON_WARMUP_CONC,
            )
            grace = scaled
        overhead = _AGENTX_NON_WARMUP_OVERHEAD_SEC + grace
        if not _is_valid_override("AGENTX_WARMUP_GRACE_PERIOD"):
            # Nothing has been tuned for this model at all. The derivation
            # above is only as good as its warmup bound, and at the canonical
            # grace that bound is the GLM-5.2/Qwen3.8 measurement (4774s/6676s
            # warmup+compile). A raw aiperf run against Kimi-K3 (conc=64, ISL
            # ~115k avg) measured warmup alone draining in ~12075s -- past this
            # whole cap before profiling starts. Nothing here can tell such a
            # model apart, so say so rather than let the round be killed
            # mid-warmup by a cap nobody chose.
            log.warning(
                "agentx_baseline_timeout_sec: neither AGENTX_BASELINE_OVERHEAD_SEC nor "
                "AGENTX_WARMUP_GRACE_PERIOD is set, so the overhead falls back to the "
                "canonical %ds (= %ds non-warmup + %ds canonical warmup grace). That "
                "grace is calibrated on GLM-5.2/Qwen3.8 and may be far too small for a "
                "long-context or slow-prefill model -- a raw aiperf run against Kimi-K3 "
                "at concurrency=64 measured warmup alone taking ~12075s. Raise "
                "AGENTX_WARMUP_GRACE_PERIOD (the client honours it too, so the warmup "
                "and this cap stay consistent) or pin AGENTX_BASELINE_OVERHEAD_SEC.",
                overhead,
                _AGENTX_NON_WARMUP_OVERHEAD_SEC,
                AGENTX_CANON_WARMUP_GRACE_SEC,
            )
    duration = _int("AGENTX_DURATION", AGENTX_DEFAULT_DURATION_SEC)
    total = duration + overhead
    # Log every input, so a timeout in the field can be read back to the value
    # that produced it instead of guessing which knob was in play.
    log.info(
        "agentx_baseline_timeout_sec: %ds = duration %ds + overhead %ds (%s)",
        total,
        duration,
        overhead,
        "explicit AGENTX_BASELINE_OVERHEAD_SEC"
        if grace is None
        else f"{_AGENTX_NON_WARMUP_OVERHEAD_SEC}s non-warmup + {grace}s warmup grace",
    )
    return total


# Cold-start settings and probes live in ``_aiter_jit`` and are re-exported
# above for callers/tests that import them from this module.


# Underscore-prefixed aliases re-exported for callers/tests; canonical
# names live in `_workload_envs`.
_default_baseline_config = default_baseline_config
_materialize_config_with_envs = materialize_config_with_envs


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
    """Resolve the benchmark ``$RESULT_DIR`` exactly as the subprocess sees it.

    Magpie is launched with ``cwd=output_dir``. If orchestration supplies a
    relative ``result_dir``, both the subprocess and Hyperloom's accuracy parser
    must interpret it relative to that same directory, not relative to the
    coordinator's process cwd.
    """
    if not override_result_dir:
        return output_dir
    result_dir = Path(override_result_dir)
    if result_dir.is_absolute():
        return result_dir
    return (output_dir / result_dir).resolve()


def _should_establish_quality_ref(task_kind: str | None, params: dict[str, Any] | None = None) -> bool:
    """Only a genuine ``baseline`` task may establish/overwrite the quality reference.

    ``replay_warm_recipe`` reuses this executor but is an optimization
    candidate, so it must compare against the pure baseline reference rather
    than redefine it (otherwise the gate would mask the warm recipe's own
    deviation from the baseline output).

    The kernel lane also drives this executor through synthetic tasks that
    carry ``kind="baseline"`` literally (integrate re-baseline, stack
    validation). Those are throughput-only A/B probes against an
    already-anchored baseline -- they never anchor one -- so they opt out via
    ``params["quality_ref_exempt"]`` and are treated exactly like
    ``replay_warm_recipe``: compare, never establish.

    Args:
        task_kind: The task kind (``ctx.task.kind``); ``None`` is treated as
            "not a baseline".
        params: The task params (``ctx.task.params``); a truthy
            ``quality_ref_exempt`` disqualifies an otherwise-genuine baseline.

    Returns:
        bool: ``True`` only for a ``"baseline"`` task that is not exempt.
    """
    if str(task_kind or "") != "baseline":
        return False
    return not (params or {}).get("quality_ref_exempt")


# Above this cold-start delta the measured round is unlikely to have settled
# either: the observed pathological case climbed 14,202 -> 19,374 -> 22,425
# tok/s across three rounds of one unchanged config, i.e. +36% into round 2 and
# another +16% into round 3.
_COLD_START_DELTA_WARN_PCT = 25.0


def _is_double_run_accuracy_handoff(
    result: dict[str, Any],
    salvaged: dict[str, Any] | None,
) -> bool:
    """Whether accuracy came from the warmup round because that is the design.

    The cold-start guard splits one baseline into a warmup round that measures
    accuracy and a measured round that measures hot throughput only, then
    decides on the measured round -- which by construction has no accuracy of
    its own. Reading the warmup round's score there is the intended handoff,
    not a recovery, and logging it as one makes every healthy double-run
    baseline look like it survived a fault.

    Args:
        result (dict[str, Any]): The deciding round's result dict.
        salvaged (dict[str, Any] | None): The salvage record, whose
            ``source_file`` names the round the accuracy came from.

    Returns:
        bool: ``True`` only for measured-round decision + warmup-round source.
    """
    out_dir = str((result or {}).get("output_dir") or "")
    if not out_dir or Path(out_dir).name != _MEASURE_ROUND_DIR:
        return False
    source = str((salvaged or {}).get("source_file") or "")
    return _WARMUP_ROUND_DIR in Path(source).parts


# Filesystem types that can be revoked / unmounted mid-run (e.g. a wekafs/NFS
# mount flap), where a process whose cwd lives on such a mount sees relative-path
# writes ENOENT. Such FS types trigger local mirroring of the InferenceX
# checkout (the server's cwd for its cuda-graph pickle dump).
_NETWORK_FS_TYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "lustre",
        "glusterfs",
        "ceph",
        "fuse.weka",
        "wekafs",
        "wekafsgw",
        "fuse.juicefs",
        "fuse.s3fs",
        "fuse.sshfs",
        "9p",
    }
)


def _path_fstype(path: str) -> str:
    """Return the filesystem type backing ``path`` per ``/proc/mounts``.

    Picks the longest mountpoint that is a prefix of the resolved path.
    Returns ``""`` when it cannot be determined (non-Linux, unreadable
    ``/proc/mounts``, ...), which callers treat as "assume local".

    Args:
        path: Filesystem path whose backing mount type is resolved.

    Returns:
        The filesystem type backing ``path``, or ``""`` when it cannot be
        determined.
    """
    try:
        rp = os.path.realpath(path)
    except OSError:
        return ""
    best_mp = ""
    best_type = ""
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                # /proc/mounts octal-escapes spaces in the mountpoint.
                try:
                    mp = parts[1].encode("latin-1").decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    mp = parts[1]
                fstype = parts[2]
                norm = mp.rstrip("/") or "/"
                if norm == "/":
                    is_under = True  # root matches everything (lowest priority)
                else:
                    is_under = rp == norm or rp.startswith(norm + "/")
                if is_under and len(norm) >= len(best_mp):
                    best_mp = norm
                    best_type = fstype
    except OSError:
        return ""
    return best_type


def _is_network_fs(path: str) -> bool:
    """True when ``path`` is backed by a revocable network filesystem.

    Args:
        path: Filesystem path to classify.

    Returns:
        ``True`` when ``path`` lives on a known network filesystem type.
    """
    return _path_fstype(path).lower() in _NETWORK_FS_TYPES


def _ensure_local_inferencex(src: str, *, mirror_key: str = "") -> str:
    """Mirror an InferenceX checkout onto stable local disk.

    Returns a local-disk path Magpie can ``cd`` into so the sglang server's
    relative-path cuda-graph snapshot dump survives a network-mount flap. No-op
    (returns ``src`` unchanged) when:

    * relocation is disabled via
      ``INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX=1``,
    * ``src`` already lives on a local filesystem, or
    * the copy fails for any reason.

    Best-effort — never raises; on failure it falls back to ``src``.
    ``mirror_key`` isolates long-running tasks that share the same source
    checkout so a later overlapping task cannot ``rmtree`` a mirror another
    server is still ``cd``-ed into.

    The caller relocates BEFORE config materialization and passes the returned
    path explicitly into materialization, so the ProfileExecutor patch step
    patches the local mirror in place.

    Args:
        src: Source InferenceX checkout path (typically on a network mount).
        mirror_key: Optional key isolating concurrent tasks that share the
            same source checkout; folded into the mirror destination name.

    Returns:
        A local-disk mirror path when relocation succeeds, otherwise ``src``
        unchanged (relocation disabled, already local, or copy failed).
    """
    src = str(src)
    if (
        os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX",
            "",
        ).strip()
        == "1"
    ):
        return src
    try:
        if not _is_network_fs(src):
            return src
    except Exception:  # noqa: BLE001 — detection is best-effort
        return src

    real_src = os.path.realpath(src)
    local_root = Path(
        os.environ.get("INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT", "")
        or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "hyperloom",
            "inferencex_local",
        )
    )
    src_hash = hashlib.sha1(real_src.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    key_hash = hashlib.sha1(str(mirror_key or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    dest_name = src_hash if not mirror_key else f"{src_hash}-{key_hash}"
    dest = local_root / dest_name
    try:
        local_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "baseline_executor: could not create local InferenceX root %s (%s); using the network-mount checkout.",
            local_root,
            exc,
        )
        return src
    # Lock keyed on dest so concurrent tasks mirroring the same source
    # serialize their rmtree/replace instead of racing.
    lock_path = str(local_root / f".{dest.name}.lock")
    staging: Path | None = None
    try:
        with best_effort_file_lock(lock_path, label="baseline_executor: InferenceX mirror lock"):
            staging = Path(tempfile.mkdtemp(dir=str(local_root)))
            staged_ix = staging / "InferenceX"
            # Copy the tree fresh every run because the per-task patch step
            # rewrites the mirror in place. Holding the lock across
            # rmtree+replace stops a concurrent task swapping ``dest`` out.
            shutil.copytree(real_src, staged_ix, symlinks=True)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(staged_ix, dest)
    except OSError as exc:
        log.warning(
            "baseline_executor: could not mirror InferenceX %s to local disk "
            "(%s); using the network-mount checkout. The #523 cuda-graph "
            "pickle dump may ENOENT if the mount flaps mid-run.",
            real_src,
            exc,
        )
        return src
    finally:
        # Always clear the staging dir so it doesn't accumulate across runs.
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if not (dest / "benchmarks" / "benchmark_lib.sh").is_file():
        log.warning(
            "baseline_executor: local InferenceX mirror at %s is incomplete; using original %s",
            dest,
            real_src,
        )
        shutil.rmtree(dest, ignore_errors=True)
        return src
    log.info(
        "baseline_executor: #523 — mirrored InferenceX from network mount %s "
        "to local disk %s so the server cwd (cuda-graph pickle dump target) "
        "survives a wekafs/NFS flap.",
        real_src,
        dest,
    )
    return str(dest)


def _git_head_sha(repo_path: str) -> str:
    """Return the current HEAD sha of a git repo, or empty string on failure."""
    if not repo_path:
        return ""
    try:
        result = subprocess.run(
            ["git", *safe_directory_args(["rev-parse", "HEAD"], cwd=repo_path)],
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
    manifest = snapshot_manifest
    if isinstance(manifest, (str, Path)):
        try:
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            result = {"ok": False, "errors": [f"manifest_read:{exc}"]}
            log.warning(
                "baseline_executor: exact patch restore failed: %s",
                result["errors"],
            )
            return result
    if not isinstance(manifest, dict):
        result = {"ok": False, "errors": ["missing_manifest"]}
        log.warning(
            "baseline_executor: exact patch restore failed: %s",
            result["errors"],
        )
        return result
    manifest_repo_value = str(manifest.get("repo_path") or "").strip()
    if not manifest_repo_value:
        result = {"ok": False, "errors": ["missing_manifest_repo"]}
        log.warning(
            "baseline_executor: exact patch restore failed: %s",
            result["errors"],
        )
        return result
    try:
        caller_repo = Path(repo_path).resolve(strict=True)
        manifest_repo = Path(manifest_repo_value).resolve(strict=True)
    except (OSError, ValueError) as exc:
        result = {
            "ok": False,
            "errors": [f"repo_validation:{type(exc).__name__}:{exc}"],
        }
        log.warning(
            "baseline_executor: exact patch restore failed: %s",
            result["errors"],
        )
        return result
    if caller_repo != manifest_repo:
        result = {
            "ok": False,
            "errors": [f"repo_mismatch:caller={caller_repo}:manifest={manifest_repo}"],
        }
        log.warning(
            "baseline_executor: exact patch restore failed: %s",
            result["errors"],
        )
        return result
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
            result = {
                "ok": False,
                "errors": [f"head_validation:{type(exc).__name__}:{exc}"],
            }
            log.warning(
                "baseline_executor: exact patch restore failed: %s",
                result["errors"],
            )
            return result
        if head != pre_sha:
            result = {
                "ok": False,
                "errors": [f"head_mismatch:expected={pre_sha}:actual={head}"],
            }
            log.warning(
                "baseline_executor: exact patch restore failed: %s",
                result["errors"],
            )
            return result
    result = _restore_patch_snapshot(manifest)
    if not result["ok"]:
        log.warning("baseline_executor: exact patch restore failed: %s", result["errors"])
    return result


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
        unmerged = subprocess.run(
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return False, f"post_3way_check_failed:{type(exc).__name__}"
    after_unmerged = unmerged.stdout.decode(errors="replace").splitlines()
    if set(after_unmerged) - set(before.get("unmerged") or []):
        return False, "new_unmerged_index_entries"
    root = Path(repo_path).resolve()
    markers = (b"<<<<<<< ", b"=======", b">>>>>>> ")
    for rel in touched:
        prior = (before.get("paths") or {}).get(rel) or {}
        if (root / f"{rel}.rej").exists() and not prior.get("reject"):
            return False, f"new_reject_file:{rel}.rej"
        target = root / rel
        marker_lines: list[str] = []
        if target.is_file() and not target.is_symlink():
            marker_lines = [
                line.decode(errors="replace") for line in target.read_bytes().splitlines() if line.startswith(markers)
            ]
        if set(marker_lines) - set(prior.get("markers") or []):
            return False, f"new_conflict_marker:{rel}"
    return True, ""


def _patch_texts_from_warm_params(params: dict[str, Any]) -> list[str]:
    """Collect readable diff text from warm-replay patch payloads."""
    patch_texts: list[str] = []
    for patch in params.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        content = str(patch.get("patch_content") or "")
        patch_ref = str(patch.get("patch_ref") or "")
        if not content and patch_ref:
            try:
                content = Path(patch_ref).read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
        if content:
            patch_texts.append(content)
    return patch_texts


def _resolve_recipe_patch_target(params: dict[str, Any]) -> str:
    """Return the framework root whose tree holds the warm-replay patch targets.

    A Recipe records the root each patch was applied into. Records written before
    that field existed carry none, and only those fall back to probing the
    patch text against the allowlist.
    """
    if not params.get("patches"):
        return ""
    from ...framework.paths import resolve_source_file_allowlist
    from .integrate_patch import _resolve_framework_root, allowlisted_explicit_root

    recorded = {
        str(entry.get("framework_root") or "").strip()
        for entry in params["patches"]
        if isinstance(entry, dict) and str(entry.get("framework_root") or "").strip()
    }
    if len(recorded) == 1:
        sole = recorded.pop()
        allowed = allowlisted_explicit_root(sole, allowlist=resolve_source_file_allowlist())
        if allowed is not None:
            return str(allowed)
        reason = f"recorded apply root {sole!r} is not usable"
    elif recorded:
        reason = f"patches record {len(recorded)} apply roots"
    else:
        reason = "no recorded apply root"

    log.info("warm replay: %s; resolving from patch targets", reason)
    root = _resolve_framework_root(
        resolve_session_framework_root() or None,
        patch_texts=_patch_texts_from_warm_params(params),
    )
    return str(root or "")


def _revert_warm_patch_state(
    target_repo: str,
    *,
    pre_sha: str = "",
    snapshot_manifest: Any = None,
    nogit_backups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Restore warm-replay patch mutations via git snapshot or nogit backups."""
    if nogit_backups:
        from ._nogit_patch import _revert_patches_no_git

        ok, errors = _revert_patches_no_git(list(nogit_backups))
        return {"ok": ok, "errors": errors, "channel": "nogit"}
    return _revert_patches(target_repo, pre_sha, snapshot_manifest)


def _apply_warm_patches(
    params: dict[str, Any],
    target_repo: str,
    output_dir: Path,
    *,
    before_mutation: Any = None,
) -> list[dict[str, str]] | dict[str, Any]:
    """Apply warm-replay code patches to the Session's active framework root.

    Reads ``params["patches"]`` (list of dicts with patch_file/patch_content/
    patch_ref). Applies each patch via ``git apply`` when the target is a git
    work-tree, otherwise via the shared nogit ``patch`` CLI path used by
    integrate_patch.

    Legacy patch lists return the list of successfully applied patch metadata
    dicts (best-effort skip semantics). Current-contract timelines set
    ``required_patch_timeline`` and fail closed: the patches are sequential, the
    first failure stops the sequence and restores the starting tree, and the
    return is a structured result dict describing the failure.
    """
    patches = params.get("patches") or []
    required_timeline = bool(params.get("required_patch_timeline"))
    if not patches:
        return []
    if not target_repo:
        if required_timeline:
            return {
                "required": True,
                "status": "failed",
                "patches": [],
                "applied": [],
                "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                "failure": "missing_target_repo",
                "rolled_back": True,
            }
        return []

    applied: list[dict[str, str]] = []
    statuses: list[dict[str, Any]] = []
    patch_log_dir = output_dir / "warm_patches"
    patch_log_dir.mkdir(parents=True, exist_ok=True)
    from ._nogit_patch import _apply_patch_no_git, _is_git_tree

    target_path = Path(target_repo)
    git_tree = _is_git_tree(target_path)
    pre_sha = _git_head_sha(target_repo) if git_tree else ""
    # prelude promotes a required timeline's tree only against a pre_sha and a
    # git snapshot manifest. nogit produces neither, so serving this path from it
    # turned a successful replay into validated_recipe_checkout_incomplete --
    # worse than the fast failure it replaced. Refuse up front, as before; nogit
    # serves the legacy list, where nothing downstream needs a sha.
    if required_timeline and not pre_sha:
        return {
            "required": True,
            "status": "failed",
            "patches": [],
            "applied": [],
            "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
            "failure": "missing_git_head",
            "pre_sha": "",
            "target_repo": target_repo,
            "rolled_back": False,
        }
    use_nogit = not git_tree or not pre_sha
    nogit_backups: list[dict[str, Any]] = []
    from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree

    resolved_contents: dict[int, str] = {}
    snapshot_contents: list[str] = []
    for idx, patch in enumerate(patches):
        patch_file = str(patch.get("patch_file") or "")
        content = str(patch.get("patch_content") or "")
        patch_ref = str(patch.get("patch_ref") or "")
        if not content and patch_ref:
            try:
                content = Path(patch_ref).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                content = ""
        reason = ""
        if not content:
            reason = "missing_artifact"
        elif not is_unified_diff(content) or "GIT binary patch" in content:
            reason = "unsafe_or_non_text_diff"
        elif patch_escapes_tree(content) is not None:
            reason = "path_escapes_tree"
        elif not _patch_touched_paths(content):
            reason = "missing_touched_paths"
        if reason:
            if required_timeline:
                return {
                    "required": True,
                    "status": "failed",
                    "patches": [{"patch_ref": patch_file, "status": "failed", "reason": reason}],
                    "applied": [],
                    "failed_ref": patch_file,
                    "failure": reason,
                    "pre_sha": pre_sha,
                    "target_repo": target_repo,
                    "rolled_back": False,
                }
            continue
        resolved_contents[idx] = content
        snapshot_contents.append(content)
    snapshot_manifest: dict[str, Any] | None = None
    if snapshot_contents and not use_nogit:
        try:
            snapshot_manifest = _create_patch_snapshot(
                target_repo,
                snapshot_contents,
                output_dir,
            )
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if required_timeline:
                return {
                    "required": True,
                    "status": "failed",
                    "patches": [],
                    "applied": [],
                    "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                    "failure": f"snapshot_failed:{type(exc).__name__}",
                    "pre_sha": pre_sha,
                    "target_repo": target_repo,
                    "rolled_back": False,
                }
            log.warning(
                "baseline_executor: skipping legacy warm patches because the "
                "rollback snapshot could not be created: %s",
                exc,
            )
            return []
        if snapshot_manifest is not None:
            params["_warm_patch_snapshot_manifest"] = snapshot_manifest
            if before_mutation is not None and not bool(before_mutation(snapshot_manifest)):
                return {
                    "required": required_timeline,
                    "status": "failed",
                    "patches": [],
                    "applied": [],
                    "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                    "failure": "pending_state_persist_failed",
                    "pre_sha": pre_sha,
                    "target_repo": target_repo,
                    "snapshot_manifest": snapshot_manifest,
                    "rollback": {"ok": True, "errors": []},
                    "rolled_back": True,
                }
    failed_ref = ""
    failure = ""

    for idx, patch in enumerate(patches):
        patch_file = patch.get("patch_file") or ""
        patch_content = resolved_contents.get(idx) or patch.get("patch_content") or ""
        patch_ref = patch.get("patch_ref") or ""
        status: dict[str, Any] = {
            "patch_ref": patch_file,
            "timeline_index": patch.get("timeline_index", idx),
        }

        if not patch_content and not patch_ref:
            log.warning(
                "baseline_executor: patch entry has no content/ref, skipping: %s",
                patch_file,
            )
            status.update(status="failed", reason="missing_artifact")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "missing_artifact"
                break
            continue

        # Resolve patch content: prefer inline content, fallback to patch_ref file.
        if not patch_content and patch_ref:
            ref_path = Path(patch_ref)
            if ref_path.is_file():
                try:
                    patch_content = ref_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.warning(
                        "baseline_executor: cannot read patch_ref %s: %s",
                        patch_ref,
                        exc,
                    )
                    status.update(status="failed", reason="artifact_read_failed")
                    statuses.append(status)
                    if required_timeline:
                        failed_ref, failure = patch_file, "artifact_read_failed"
                        break
                    continue
            else:
                log.warning(
                    "baseline_executor: patch_ref not found: %s",
                    patch_ref,
                )
                status.update(status="failed", reason="missing_artifact")
                statuses.append(status)
                if required_timeline:
                    failed_ref, failure = patch_file, "missing_artifact"
                    break
                continue

        # Structural safety gate on untrusted KB-sourced patch_content before
        # it is git-applied to the live checkout: reject non-diff blobs and any
        # patch whose header path escapes the tree (absolute / ``..``). Stale /
        # missing-target patches are left to git apply's own check so a
        # legitimate warm patch is never dropped here.
        if not is_unified_diff(patch_content) or "GIT binary patch" in patch_content:
            log.warning(
                "baseline_executor: skipping warm patch %s — not a unified diff",
                patch_file,
            )
            status.update(status="failed", reason="unsafe_or_non_text_diff")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "unsafe_or_non_text_diff"
                break
            continue
        _escape = patch_escapes_tree(patch_content)
        if _escape is not None:
            log.warning(
                "baseline_executor: skipping warm patch %s — path escapes tree: %r",
                patch_file,
                _escape,
            )
            status.update(status="failed", reason="path_escapes_tree")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "path_escapes_tree"
                break
            continue

        # Write patch to temp file then apply.
        patch_path = patch_log_dir / f"{idx:03d}_{Path(patch_file).stem or 'patch'}.diff"
        patch_path.write_text(patch_content, encoding="utf-8")

        method = ""
        try:
            if use_nogit:
                backup_root = patch_log_dir / "patch_backups"
                ok, err, backups, _feedback = _apply_patch_no_git(
                    target_path,
                    patch_path,
                    backup_root,
                    seq_offset=len(nogit_backups),
                )
                if not ok:
                    raise RuntimeError(err or "nogit patch apply failed")
                nogit_backups.extend(backups)
                method = "applied_nogit"
            else:
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
                    method = "applied"
                elif required_timeline:
                    reverse = subprocess.run(
                        ["git", "apply", "-R", "--check", str(patch_path)],
                        cwd=target_repo,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    if reverse.returncode == 0:
                        method = (
                            "already_present"
                            if _patch_present_in_committed_head(
                                target_repo,
                                patch_path,
                            )
                            else "present_in_dirty_worktree"
                        )
                    else:
                        touched = _patch_touched_paths(patch_content)
                        before_residue = _three_way_residue_snapshot(
                            target_repo,
                            touched,
                        )
                        three_way = subprocess.run(
                            ["git", "apply", "--3way", str(patch_path)],
                            cwd=target_repo,
                            capture_output=True,
                            timeout=30,
                            check=False,
                        )
                        if three_way.returncode == 0:
                            clean, residue = _verify_three_way_clean(
                                target_repo,
                                touched,
                                before_residue,
                            )
                            if not clean:
                                raise RuntimeError(residue)
                            method = "applied_3way"
                        else:
                            detail = (
                                three_way.stderr.decode(errors="replace")[:500]
                                if three_way.stderr
                                else "git apply --3way failed"
                            )
                            raise RuntimeError(detail)
                else:
                    detail = (
                        checked.stderr.decode(errors="replace")[:500] if checked.stderr else "git apply --check failed"
                    )
                    raise RuntimeError(detail)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            log.warning(
                "baseline_executor: warm patch apply failed for %s: %s",
                patch_file,
                exc,
            )
            reason = "nogit_apply_failed" if use_nogit else "git_apply_failed"
            status.update(status="failed", reason=reason, detail=str(exc)[:500])
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, reason
                break
            continue

        item = {
            "patch_file": patch_file,
            "idx": str(idx),
            "status": method,
        }
        applied.append(item)
        status["status"] = method
        statuses.append(status)

    if nogit_backups:
        params["_warm_patch_nogit_backups"] = nogit_backups

    if required_timeline:
        if failed_ref:
            restore = _revert_warm_patch_state(
                target_repo,
                pre_sha=pre_sha,
                snapshot_manifest=snapshot_manifest,
                nogit_backups=nogit_backups,
            )
            return {
                "required": True,
                "status": "failed",
                "patches": statuses,
                "applied": applied,
                "failed_ref": failed_ref,
                "failure": failure,
                "pre_sha": pre_sha,
                "target_repo": target_repo,
                "snapshot_manifest": snapshot_manifest,
                "rolled_back": bool(restore.get("ok")),
                "rollback": restore,
            }
        return {
            "required": True,
            "status": "prepared",
            "patches": statuses,
            "applied": applied,
            "failed_ref": "",
            "pre_sha": pre_sha,
            "target_repo": target_repo,
            "snapshot_manifest": snapshot_manifest,
            "rolled_back": False,
        }
    return applied


def _rollback_warm_kernel_apply_results(
    results: Any,
    snapshots: Any = None,
) -> dict[str, Any]:
    """Rollback kernel mutations and report whether every restore succeeded.

    Restoring the snapshots is not on its own enough: the applies also left
    backup manifests, and on the multi-node path the patch is on every pod.
    Both halves have to be undone, which is what the prelude rollback does.

    Args:
        results (Any): Apply results carrying the backup manifests to revert.
        snapshots (Any): Pristine copies captured before the mutation.

    Returns:
        dict[str, Any]: ``ok`` and the accumulated ``errors``.
    """
    if not isinstance(results, list):
        return {"ok": False, "errors": ["invalid_apply_results"]}
    from ...phases.prelude import PreludePhase

    return PreludePhase._revert_warm_kernel_patches(
        [r for r in results if isinstance(r, dict)],
        list(snapshots) if isinstance(snapshots, list) else None,
    )


class BaselineExecutor:
    """Class form for tests / DI; ``baseline_executor`` is the bare callable.

    ``session_dir`` is the session root for the per-task workspace
    (``<sd>/runs/baseline/<task_id>/``); used only as a fallback when the
    SubAgentRunner injects a pre-created workspace via ``ctx.extra``.
    """

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
        default_timeout_sec: int = BASELINE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str | None = None,
    ):
        """Initialize the baseline executor with launch defaults.

        Args:
            magpie_python (str | None): Python interpreter used to invoke
                Magpie; resolved automatically when ``None``.
            default_config_path (Path | str | None): Default Magpie YAML config
                path; resolved from ``$FRAMEWORK`` at call time when ``None``.
            session_dir (Path | str | None): Canonical session root for
                per-task workspaces; resolved automatically when ``None``.
            shared_state: Optional live SharedState object. When provided, the
                eager-fallback one-shot is consumed in memory before saving so
                Coordinator cannot later re-persist a stale True value.
            default_timeout_sec (int): Default (warm-start) subprocess timeout.
            cwd (Path | str | None): Working directory for the Magpie subprocess.
        """
        from ._grid_runner import _resolve_session_dir

        # Backend-aware interpreter: bypass uses a plain python3, magpie
        # uses the Magpie-importable venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        self.magpie_python = magpie_python or resolve_benchmark_interpreter()
        # None = resolve from $FRAMEWORK at call time; explicit fixture path wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.shared_state = shared_state
        self.default_timeout_sec = default_timeout_sec
        self.cwd = Path(cwd if cwd is not None else tempfile.gettempdir())

    def _resolve_default_config(self) -> Path:
        """Hook for subclasses (ProfileExecutor) to swap the resolver.

        Returns:
            Path: The default baseline Magpie YAML config path.
        """
        return _default_baseline_config()

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        """Pick the per-task workspace dir.

        Order: ``task.params['output_dir']`` → ``ctx.extra['workspace']``
        → ``runs_dir(...)`` (direct-instantiation fallback for tests).

        Args:
            ctx: Runner context carrying ``task.params`` and ``extra``.
            action: Action name used when falling back to ``runs_dir(...)``.

        Returns:
            The resolved per-task workspace directory.
        """
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, action, ctx.task.task_id)

    def _resolve_shared_state(self, shared_state: Any | None = None) -> Any:
        """Resolve the live SharedState for a session-scoped flag read/write.

        Args:
            shared_state: Optional live SharedState; falls back to
                ``self.shared_state`` and then a loaded session state.

        Returns:
            The resolved SharedState instance.
        """
        state = shared_state or self.shared_state
        if state is None:
            from ...state.shared_state import SharedState

            state = SharedState.load_or_init(self.session_dir)
        return state

    def _eager_fallback_armed(self, shared_state: Any | None = None) -> bool:
        """Peek the one-shot eager fallback flag WITHOUT consuming it.

        Used to keep the flag armed when the framework is unknown (cannot pick
        a safe disable-cuda-graph flag), so the one-shot is not wasted.
        Best-effort: missing/unreadable state reads as not armed.

        Args:
            shared_state: Optional live SharedState; falls back to
                ``self.shared_state`` and then a loaded session state.

        Returns:
            ``True`` when the one-shot eager-fallback flag is currently armed.
        """
        try:
            state = self._resolve_shared_state(shared_state)
            return bool(getattr(state, "baseline_eager_fallback", False))
        except Exception:  # noqa: BLE001 — fallback must never break baseline
            log.debug(
                "baseline_executor: eager-fallback flag peek failed",
                exc_info=True,
            )
            return False

    def _consume_eager_fallback(self, shared_state: Any | None = None) -> bool:
        """Consume the one-shot cuda-graph eager fallback flag from SharedState.

        Returns True (and clears the flag) when a prior baseline armed it.
        Best-effort: missing/unreadable state reads as no fallback.

        Args:
            shared_state: Optional live SharedState; falls back to
                ``self.shared_state`` and then a loaded session state.

        Returns:
            ``True`` when the flag was armed (and is now cleared), else
            ``False``.
        """
        try:
            state = self._resolve_shared_state(shared_state)
            if not getattr(state, "baseline_eager_fallback", False):
                return False
            state.baseline_eager_fallback = False
            state.save(self.session_dir)
            return True
        except Exception:  # noqa: BLE001 — fallback must never break baseline
            log.debug(
                "baseline_executor: eager-fallback flag check failed",
                exc_info=True,
            )
            return False

    def _resolve_timeout(self, params: dict[str, Any]) -> int:
        """Pick the subprocess timeout for this baseline launch.

        Order: explicit ``task.params['timeout_sec']`` → cold-start cap
        when the aiter jit probe reports COLD (env-overridable via
        ``INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC``) → warm default.
        Every path emits one log line for greppability.

        All three are hang backstops sized in hours, with no reference to the
        session budget; :meth:`_session_capped_timeout` reduces the result to
        what is actually left, per round, at the point each round launches.

        Args:
            params: Task params; an explicit ``timeout_sec`` overrides the
                probe-based selection.

        Returns:
            The subprocess timeout in seconds for this baseline launch.
        """
        explicit = params.get("timeout_sec")
        if explicit:
            timeout_sec = int(explicit)
            log.info(
                "baseline_executor: timeout=%ds (explicit task param)",
                timeout_sec,
            )
            return timeout_sec

        # Ahead of the probe on purpose: the probe cannot answer this case. See
        # AGENTX_BASELINE_OVERHEAD_SEC -- AgentX moves max_model_len, which is
        # part of the JIT signature the cold/warm split is really about, while
        # the probe only counts kernels globally. Asking it first would return
        # WARM and the 7800s cap that a first AgentX round cannot meet.
        if agentx_enabled():
            timeout_sec = agentx_baseline_timeout_sec()
            log.info(
                "baseline_executor: timeout=%ds (AgentX: AGENTX_DURATION + overhead). "
                "The aiter cold/warm probe is not consulted -- it counts kernels "
                "globally and cannot see that AgentX changes the JIT signature, so "
                "it reports WARM while the round pays a first-compile.",
                timeout_sec,
            )
            return timeout_sec

        cache = _probe_aiter_jit_cache()
        cold_cap = int(
            os.environ.get(
                "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
                BASELINE_COLD_START_TIMEOUT_SEC,
            )
        )
        if cache["probe_status"] == "found" and cache["is_cold"]:
            # Before paying the cold-start compile, reap aiter JIT locks left by
            # a killed hipcc. Gated on "no live compiler" so a concurrent
            # benchmark's in-flight compile is never disturbed.
            sweep = sweep_stale_aiter_locks_if_dead()
            if sweep.get("skipped_live"):
                log.info(
                    "baseline_executor: aiter lock sweep skipped — live "
                    "compiler process present (jit dir node-shared).",
                )
            elif sweep.get("deleted"):
                log.warning(
                    "baseline_executor: reaped %d stale aiter JIT lock(s) "
                    "under %s (compiler_alive=%s) before cold start.",
                    sweep["deleted"],
                    sweep.get("dir"),
                    sweep.get("compiler_alive"),
                )
                # Locks gone — re-probe so the log line below reflects reality.
                cache = _probe_aiter_jit_cache()
        if cache["probe_status"] == "found" and cache["is_cold"]:
            log.warning(
                "baseline_executor: COLD_START detected — aiter jit/build/ "
                "at %s has %d .so (< %d threshold), %d MB. Bumping timeout "
                "%ds -> %ds. First-time JIT compile on a new "
                "(model, dtype, TP, max_model_len) signature can take 30+ "
                "minutes for large FP8 / MoE models.",
                cache["path"],
                cache["kernel_count"],
                COLD_START_KERNEL_THRESHOLD,
                cache["size_mb"],
                self.default_timeout_sec,
                cold_cap,
            )
            return cold_cap
        if cache["probe_status"] == "found":
            log.info(
                "baseline_executor: WARM start — aiter jit/build/ at %s has %d .so, %d MB. Using default timeout=%ds.",
                cache["path"],
                cache["kernel_count"],
                cache["size_mb"],
                self.default_timeout_sec,
            )
            return self.default_timeout_sec
        log.warning(
            "baseline_executor: aiter jit cache not located "
            "(probe_status=%s). Using default timeout=%ds. Cold-start "
            "auto-bump disabled for this run.",
            cache["probe_status"],
            self.default_timeout_sec,
        )
        return self.default_timeout_sec

    @staticmethod
    def _session_capped_timeout(
        timeout_sec: int,
        session_deadline_sec: float | None,
        *,
        output_dir: Path,
    ) -> int:
        """``timeout_sec`` reduced to what the session can still pay for.

        The baseline's own timeout is a catastrophic-hang backstop -- two hours
        by default, four for a cold start -- chosen with no reference to how much
        of the session is left. A round granted more than the budget has runs
        past the end of the session and takes the closing phase with it.

        Nothing is held back here, and no pass of a baseline round holds anything
        back either. That is what keeps every cap sitting past the session
        deadline, so the watchdog reaches a round before the round's own timeout
        does and the kill is attributed to the budget rather than to the model.
        Whether a round should start, and whether its measured pass should follow
        its warmup, are decided by the gates that price those questions -- not by
        shortening a cap until the round dies of it.

        Args:
            timeout_sec: The timeout this round would get on an unbounded budget.
            session_deadline_sec: Monotonic-clock session deadline, or ``None``
                when there is no budget to respect.
            output_dir: The round's workspace, for the log line.

        Returns:
            int: The hard timeout to grant this round, in seconds.
        """
        return _logged_session_clamp(
            timeout_sec,
            session_clamped_timeout_sec(timeout_sec, session_deadline_sec),
            output_dir=output_dir,
        )

    @staticmethod
    def _inferencex_root_from_config(config_path: Path) -> str:
        """Resolve the InferenceX checkout the subprocess will ``cd`` into.

        Prefers the materialized ``benchmark.inferencex_path`` (the task-local,
        possibly local-disk-mirrored checkout) and falls back to
        ``$INFERENCEX_PATH``. Returns ``""`` when neither is set.

        Args:
            config_path: The materialized Magpie YAML config path.

        Returns:
            The InferenceX checkout path, or ``""`` when unresolved.
        """
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
        """Hook after YAML materialization, before launch.

        Applies the eval-dest redirect so ``append_lm_eval_summary``'s ``mv ./``
        writes lm-eval ``results*.json`` into ``$RESULT_DIR`` instead of the
        process cwd, which is outside the session once InferenceX is mirrored to
        local disk and would fail the accuracy gate despite a passing eval.

        The patches are re-asserted here rather than at install time because
        Magpie's ``_prepare_benchmark_scripts`` re-copies its own generic ``*.sh``
        into ``<inferencex>/benchmarks/`` and :func:`_ensure_local_inferencex`
        re-mirrors the checkout on every run, so an install-time patch does not
        survive. Materialization has pinned the exact checkout by this point.

        ProfileExecutor fully REPLACES this hook (it does not call ``super()``)
        with NUM_PROMPTS / PROFILE_EXTRA_BODY validation, so the eval patches
        below apply to the baseline path only.

        Args:
            config_path: The materialized Magpie YAML config path.
            output_dir: The per-task workspace directory.

        Returns:
            An early-return result dict to short-circuit the launch, or
            ``None`` to proceed with the baseline run.
        """
        ix_root = self._inferencex_root_from_config(config_path)
        if ix_root:
            ensure_benchmark_lib_eval_dest_patched(Path(ix_root))
            ensure_benchmark_lib_eval_start_patched(Path(ix_root))
        if not materialized_run_eval_disabled(config_path):
            # Target present but unpatchable is a hard stop; target absent is an
            # unrecognized layout, which warns rather than failing every eval run.
            probe_root = Path(ix_root) if ix_root else None
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
            # Fail LOUDLY (never warn-and-continue) when the fatal eval flag cannot
            # be removed AND this run is meant to execute lm-eval: the benchmark is
            # guaranteed to abort in run_lm_eval, and the accuracy gate then stops
            # the whole session. Surfacing it here costs seconds instead of a full
            # doomed server boot + benchmark.
            try:
                compat_ok = ensure_eval_concurrency_compat(inferencex_dir=ix_root or None)
            except Exception as exc:  # noqa: BLE001 — never mask as a silent skip
                log.error(
                    "baseline_executor: eval-concurrency compat patch raised for %s: %s",
                    ix_root,
                    exc,
                )
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
            anchor_result = self._eval_patch_anchors_result(ix_root)
            if anchor_result is not None:
                return anchor_result
        return None

    # Scoped to the line-replacement patches this hook applies. The probe is
    # judged above by its own target-exists test, and ``num_prompts`` /
    # ``profile_extra_body`` belong to the profiling path and are
    # ProfileExecutor's to judge.
    _EVAL_HOOK_ANCHORS = ("eval_dest", "eval_start")
    # Of those, the one whose silent absence corrupts the accuracy gate rather
    # than degrading it: without ``eval_dest`` the results file lands in the cwd
    # and no score is ever parsed. ``eval_start`` is a log breadcrumb -- worth
    # reporting, never worth aborting for.
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

        Args:
            ix_root: The resolved InferenceX root for this run, if any.

        Returns:
            An early-return failure dict, or ``None`` to proceed.
        """
        broken = [s for s in failed_patch_anchors(ix_root or None) if s.name in self._EVAL_HOOK_ANCHORS]
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

    @staticmethod
    def _failure_carries_markers(
        result: dict[str, Any],
        markers: tuple[str, ...],
        context_markers: tuple[str, ...] | None = None,
    ) -> bool:
        """Whether a failed result's error tail, warnings or logs hit a marker.

        Scans the result's ``error`` tail + ``nonfatal_warnings`` and then any
        benchmark stdout/stderr + ``server.log`` under the run's ``output_dir``.
        Recursive so the double-run path is covered (the warmup round's logs
        carry the marker even when the measure round failed for a downstream
        reason). Never raises.

        Args:
            result: A ``status="failed"`` baseline result dict.
            markers: Substrings identifying the root cause being probed.
            context_markers: When set, one of these must appear in the SAME
                blob as a ``markers`` hit, narrowing a generic signature to the
                subsystem that owns it.

        Returns:
            ``True`` when the marker (and its context) is found, else ``False``.
        """

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
        # Double-run: the failure markers may live in the sibling warmup round,
        # so climb to the shared task root to scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False
        log_names = ("benchmark_stderr.log", "benchmark_stdout.log", "server.log")
        seen = 0
        try:
            for path in root.rglob("*.log"):
                if path.name not in log_names:
                    continue
                seen += 1
                if seen > 64:  # bound the scan on pathological trees
                    break
                try:
                    with path.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - _LOG_SCAN_MAX_BYTES))
                        chunk = f.read().decode("utf-8", "replace")
                except OSError:
                    continue
                if _hit(chunk):
                    return True
        except OSError:
            return False
        return False

    @staticmethod
    def _record_baseline_convergence(
        result: dict[str, Any],
        warmup_tput: Any,
    ) -> None:
        """Record how steady the baseline anchor actually is.

        The double-run discards round 1 by design, which leaves exactly one
        usable measurement -- and one measurement cannot be shown to be steady.
        That is worth stating rather than assuming: the whole gain ledger is
        graded against this number with a 3% KEEP threshold, and a real session
        once produced 14,202 -> 19,374 -> 22,425 tok/s from one unchanged
        configuration. Establishing convergence needs a third round.

        So the verdict is recorded (it will read ``insufficient_rounds``) along
        with the cold-start delta, and a warning is raised only when that delta
        is large enough to suggest round 2 had not settled either. Never raises,
        and never fails the baseline -- halting here would stall the session.
        """
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
        except Exception:  # noqa: BLE001 - observability must never break a baseline
            log.debug("baseline convergence record failed", exc_info=True)

    @staticmethod
    def _eval_failure_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        """Detect an eval-rooted baseline failure and capture bounded evidence.

        Scans the result's ``error`` tail + ``nonfatal_warnings`` and then any
        benchmark stdout/stderr + ``server.log`` under the run's ``output_dir``
        for ``run_eval``-failure markers. Climbs to the shared task root so the
        double-run warmup logs are covered. Returns the first matched window
        (marker plus a bounded slice) so callers need not rescan. Never raises.

        Returns:
            ``(is_eval_rooted, evidence)``; ``evidence`` is empty when not rooted.
        """

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
        # Double-run: the failure markers may live in the sibling warmup round,
        # so climb to the shared task root to scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False, ""
        log_names = ("benchmark_stderr.log", "benchmark_stdout.log", "server.log")
        seen = 0
        try:
            for path in root.rglob("*.log"):
                if path.name not in log_names:
                    continue
                seen += 1
                if seen > 64:  # bound the scan on pathological trees
                    break
                try:
                    with path.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - _LOG_SCAN_MAX_BYTES))
                        chunk = f.read().decode("utf-8", "replace")
                except OSError:
                    continue
                w = _window(chunk)
                if w is not None:
                    return True, f"{path.name}: {w}"
        except OSError:
            return False, ""
        return False, ""

    @staticmethod
    def _is_eval_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline result was caused by the accuracy eval.

        Args:
            result: A ``status="failed"`` baseline result dict.

        Returns:
            ``True`` when an eval-failure marker is found, else ``False``.
        """
        return BaselineExecutor._failure_carries_markers(result, _EVAL_FAILURE_MARKERS)

    @staticmethod
    def _is_moe_runner_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline died on the MoE runner backend in use.

        The quant scheme's ``runner`` is only built on the backends it actually
        implements, so an unsupported ``--moe-runner-backend`` surfaces as a
        missing-attribute crash on the first forward pass.

        Args:
            result: A ``status="failed"`` baseline result dict.

        Returns:
            ``True`` when a MoE-runner marker is found, else ``False``.
        """
        return BaselineExecutor._failure_carries_markers(
            result,
            _MOE_RUNNER_MISSING_MARKERS,
            _MOE_SCHEME_CONTEXT_MARKERS,
        )

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the Magpie baseline, with a one-shot eval-failure fallback.

        Delegates to :meth:`_run_once`. When the run fails for an eval-rooted
        reason AND accuracy eval was active, it re-runs **once** with
        ``RUN_EVAL=false`` so the throughput baseline is salvaged. The retried
        result is tagged ``accuracy_source="eval_unavailable"`` and carries a
        ``eval_failed_fallback_no_accuracy`` warning.

        **The salvage retry is skipped for a genuine ``baseline`` task.** A
        baseline's job is to establish the accuracy reference, so
        :meth:`_maybe_stop_on_missing_baseline_accuracy` halts the run whenever
        eval was expected and produced nothing — including after this very
        fallback (the fallback forces ``RUN_EVAL=false`` but eval was still
        expected and still broke). Running the retry there therefore burns a
        second full server boot + benchmark to produce a result that is
        guaranteed to be discarded, and delays the operator's error by minutes.
        Fail fast instead: record the same ``baseline_accuracy_failed`` stop
        immediately. The retry is kept for non-baseline kinds (e.g.
        ``replay_warm_recipe``), which do not establish the quality reference
        and for which a throughput-only result IS usable.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                (config / model / timeout knobs) and ``extra`` (workspace).

        Returns:
            dict[str, Any]: The baseline result dict (see :meth:`_run_once`).

        Raises:
            FileNotFoundError: If the resolved baseline config does not exist.
        """
        result = await self._run_once(ctx)
        params = ctx.task.params or {}
        # A failed required patch timeline means the donor is incompatible with
        # the current tree. The run restored Recipe + Kernel changes and the
        # result is returned as a failed warm replay; PRELUDE marks it failed
        # and the session optimizes from the clean baseline.
        # "Already off" only when the operator explicitly disabled eval — via
        # ``--no-eval``, the param, or an extra_envs RUN_EVAL that is PRESENT and
        # falsey. An absent RUN_EVAL must NOT count.
        _extra_envs = params.get("extra_envs") or {}
        _explicit_run_eval = (
            "RUN_EVAL" in _extra_envs and str(_extra_envs["RUN_EVAL"]).strip().lower() in _RUN_EVAL_FALSE_VALUES
        )
        eval_already_off = is_truthy(params.get("disable_run_eval")) or _explicit_run_eval or self._eval_disabled(ctx)
        eval_disabled_by_fallback = False
        if result.get("status") != "succeeded" and not eval_already_off and self._is_eval_rooted_failure(result):
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
            if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
                log.error(
                    "baseline_executor: failure is eval-rooted (InferenceX "
                    "run_eval aborted the benchmark) on a genuine baseline, "
                    "whose whole purpose is to establish the accuracy "
                    "reference. NOT retrying with RUN_EVAL=false: a "
                    "throughput-only baseline cannot satisfy the accuracy gate, "
                    "so the retry would burn a second full benchmark and the "
                    "run would stop anyway. Stopping now — fix the accuracy "
                    "eval (see the benchmark stdout/stderr for the run_eval "
                    "error) rather than disabling RUN_EVAL."
                )
                result.setdefault("nonfatal_warnings", [])
                result["nonfatal_warnings"].append("eval_failed_no_fallback_baseline_requires_accuracy")
                result["accuracy_source"] = "eval_unavailable"
                self._request_eval_rooted_baseline_stop(ctx, result)
                return result
            log.warning(
                "baseline_executor: failure looks eval-rooted (InferenceX "
                "run_eval aborted the benchmark); retrying once with "
                "RUN_EVAL=false to salvage the throughput baseline without "
                "the accuracy gate."
            )
            retry = await self._run_once(ctx, force_disable_eval=True)
            retry.setdefault("nonfatal_warnings", [])
            retry["nonfatal_warnings"].append("eval_failed_fallback_no_accuracy")
            if retry.get("status") == "succeeded":
                retry["accuracy_source"] = "eval_unavailable"
            eval_disabled_by_fallback = True
            result = retry
        if result.get("status") != "succeeded" and self._is_moe_runner_rooted_failure(result):
            log.warning(
                "baseline_executor: the server died on a MoE runner backend "
                "that has no implementation for this checkpoint's quant "
                "scheme; retrying once without --moe-runner-backend so the "
                "framework picks the backend itself."
            )
            # Carry the eval fallback forward: the eval that broke above must
            # stay off, and its bookkeeping must survive onto this result.
            retry = await self._run_once(
                ctx,
                force_disable_eval=eval_disabled_by_fallback,
                force_drop_moe_runner_backend=True,
            )
            retry.setdefault("nonfatal_warnings", [])
            if eval_disabled_by_fallback:
                retry["nonfatal_warnings"].append("eval_failed_fallback_no_accuracy")
                if retry.get("status") == "succeeded":
                    retry["accuracy_source"] = "eval_unavailable"
            retry["nonfatal_warnings"].append("moe_runner_backend_fallback_dropped_flag")
            result = retry
        self._maybe_stop_on_missing_baseline_accuracy(ctx, result)
        return result

    def _eval_disabled(self, ctx: RunnerContext) -> bool:
        """Whether ``--no-eval`` turned the accuracy eval off for this session."""
        extra = getattr(ctx, "extra", None) or {}
        state = self._resolve_shared_state(extra.get("shared_state"))
        return bool(getattr(state, "eval_disabled", False))

    def _eval_enablement_active(self, ctx: RunnerContext) -> bool:
        """Whether an eval failure should route into enablement this run.

        Only for a genuine baseline, single-node, with the eval lane admitted by
        the session's enablement mode.
        """
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
        """Mark ``result`` as an eval-rooted baseline failure for enablement.

        Records the failure kind, observed accuracy, effective floor, bounded
        evidence and an eval-contract fingerprint so writeback can persist the
        trigger and a later enablement round can re-run the same contract.
        """
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
        framework = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None
        model = params.get("model") or params.get("resolved_model")
        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        result[BASELINE_EVAL_FAILED_KEY] = True
        result[BASELINE_EVAL_FAILURE_KIND_KEY] = kind
        result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] = observed_accuracy
        result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] = floor
        result[BASELINE_EVAL_EVIDENCE_KEY] = (evidence or "")[:4000]
        # Fingerprint derives from the materialized YAML contract fields only —
        # task/metric are result outputs and may be absent on eval crash, so they
        # must not participate in the stable identity.
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
        """Halt the run for an eval-rooted baseline failure, fail-fast path.

        :meth:`_maybe_stop_on_missing_baseline_accuracy` only inspects
        ``succeeded`` results (a plain throughput failure is the Coordinator's
        ``baseline_failed`` streak business). An eval-rooted *failure* on a
        genuine baseline is a different animal: the accuracy reference can
        never be established, so it must produce the same
        ``baseline_accuracy_failed`` stop the post-fallback path used to
        produce — just minutes earlier and without the wasted re-benchmark.

        A sibling attempt may already have measured a valid accuracy (the
        cold-start guard and the Coordinator's retries each get their own
        ``runs/baseline/<attempt>`` dir), so the same salvage the succeeded
        path uses is attempted first.

        Args:
            ctx (RunnerContext): The runner context (task kind + shared_state).
            result (dict): The failed baseline result dict, mutated in place
                when a sibling accuracy is salvaged.
        """
        from ._accuracy_gate import accuracy_meets_floor, request_baseline_accuracy_stop

        params = ctx.task.params or {}
        framework = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            acc_val = self._apply_salvaged_accuracy(result, salvaged, shared_state)
            # ``accuracy_meets_floor`` already means "finite, strictly positive
            # and >= floor", so floor 0.0 is the legacy "any usable accuracy".
            if accuracy_meets_floor(acc_val, 0.0):
                log.warning(
                    "baseline_executor: eval-rooted baseline failure, but salvaged "
                    "a valid baseline accuracy=%.4f from a sibling attempt (%s); "
                    "not stopping the run",
                    acc_val,
                    salvaged.get("source_file", ""),
                )
                return
            # A measured zero is a broken baseline, not a usable reference: it
            # must still reach the stop below, now with the score on record.
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
        """Halt the run when a genuine baseline produced no accuracy result.

        A baseline is supposed to establish the accuracy reference. If the
        accuracy test was expected to run but produced no usable result, the
        setup is fundamentally broken and the whole run stops.

        "No usable result" means a missing or non-positive accuracy: scriptable
        workloads record ``accuracy=0.0`` (fail-closed) when the quality gate is
        absent, and serving records no accuracy at all, so both are covered.

        Incidental disabling is no opt-out. ``disable_run_eval``, an explicit
        ``RUN_EVAL=false`` env, or a YAML/reference-env value do not make a
        missing accuracy acceptable on a genuine baseline -- they only mean the
        reference was never measured, which is exactly what this guard rejects.
        Two deliberate opt-outs exist: ``quality_ref_exempt`` (synthetic
        kernel-lane re-baselines) and ``--no-eval`` (no reference was asked for).

        With eval-on-fail enablement active (the default), the result is stamped
        as an eval-failure contract -- ``eval_generation_pathology`` when the
        generation probe tripped, else whatever the score classifies as -- and
        routed to enablement rather than
        stopping; ``_is_promotable_result`` then blocks it from anchoring
        ``baseline_tput`` / ``baseline_accuracy`` / ``baseline_config_path``.
        Otherwise the run stops. Throughput-level baseline failures are handled
        by the Coordinator's existing ``baseline_failed`` streak logic.

        Args:
            ctx (RunnerContext): The runner context (task kind + shared_state).
            result (dict): The final baseline result dict.
        """
        if not _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
            return
        if self._eval_disabled(ctx):
            return
        # A failed status must NOT skip straight past the salvage below. An eval
        # that dies mid-run (e.g. the server vanishes while lm_eval is working
        # through its requests) makes run_eval exit non-zero, which fails the
        # whole round -- exactly the case the sibling-accuracy salvage exists to
        # rescue. Returning here meant the salvage could only ever run when
        # nothing had gone wrong, so a perfectly good accuracy measured by the
        # cold-start guard's first round was discarded and the run stopped.
        #
        # Throughput-level failures still fall through harmlessly: they leave no
        # results*.json for the salvage to find, so it returns None and the
        # normal stop path continues. Those remain the Coordinator's
        # ``baseline_failed`` streak logic to handle.
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
        )

        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        if eval_enablement:
            if accuracy_meets_floor(acc, floor):
                return  # a usable baseline accuracy at/above the floor exists
        elif acc is not None and float(acc) > 0.0:
            return  # a usable baseline accuracy exists
        params = ctx.task.params or {}
        framework = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip() or None
        from ._accuracy_gate import request_baseline_accuracy_stop

        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        # Session-level salvage (complements #942): the cold-start guard and the
        # coordinator's retries each run in their own ``runs/baseline/<attempt>``
        # dir. #942 keeps every attempt's eval output inside that attempt's
        # ``$RESULT_DIR``, but the accuracy-stop decision runs on the *deciding*
        # attempt -- whose dir can be empty when a prior sibling attempt already
        # produced a valid ``results*.json``. Before halting, reuse a positive
        # accuracy measured by any sibling attempt rather than discarding a good
        # baseline and stopping the whole run.
        #
        # This runs BEFORE the enablement branch below, not after it. The
        # cold-start guard splits one baseline into warmup_round (RUN_EVAL=true
        # -- the only round that measures accuracy) and measure_round
        # (RUN_EVAL=false -- hot throughput only), then decides on the
        # measure_round result, which by construction carries no accuracy. With
        # enablement active the branch below used to fire first and dispatch a
        # specialist to chase a quality regression that never happened, while
        # the sibling warmup_round held a perfectly good score (gsm8k 0.8954 on
        # Kimi-K3, 2026-07-29). Only a genuinely missing or below-floor accuracy
        # should reach enablement.
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
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
            # Floor 0.0 reproduces the non-enablement "any positive accuracy is
            # usable" rule; ``accuracy_meets_floor`` rejects zero either way.
            if accuracy_meets_floor(acc_val, floor if eval_enablement else 0.0):
                return
            # Salvaged, but unusable (zero or still under the floor): that is a
            # real quality signal, so fall through with the observed value
            # rather than reporting it as a missing measurement.
            acc = acc_val

        # Route into enablement instead of stopping: the throughput baseline
        # stays for diagnostics but is blocked from anchoring.
        if eval_enablement:
            kind = classify_accuracy_failure(acc, floor)
            observed = float(acc) if isinstance(acc, (int, float)) else None
            evidence = (
                f"baseline accuracy did not meet floor: accuracy={acc} floor={floor} "
                f"task={result.get('accuracy_task')} metric={result.get('accuracy_metric')} "
                f"source={result.get('accuracy_source')}"
            )
            # A tripped probe means the eval was cut short because the model
            # never stopped generating, not that it answered and got them wrong.
            probe = result.get("eval_probe")
            if probe:
                kind = EVAL_KIND_GENERATION_PATHOLOGY
                evidence = f"{evidence}; {eval_probe_summary(probe)}"
            self._stamp_eval_failure_contract(
                ctx, result, kind=kind or "", observed_accuracy=observed, evidence=evidence
            )
            log.warning(
                "baseline_executor: accuracy %s below floor %.4f (kind=%s); routing to "
                "enablement instead of stopping the run.",
                acc,
                floor,
                kind,
            )
            return
        request_baseline_accuracy_stop(
            shared_state,
            context=f"baseline:{framework or 'unknown'}",
        )

    def _apply_salvaged_accuracy(
        self,
        result: dict[str, Any],
        salvaged: dict[str, Any],
        shared_state: Any,
        *,
        expected_handoff: bool = False,
    ) -> float:
        """Record a salvaged sibling accuracy, publishing it as the gate
        reference only when it can serve as one.

        ``result`` carries the score whatever its value: that is evidence.
        ``SharedState.baseline_accuracy`` is the reference later gates compare
        against, where ``<= 0`` is :func:`accuracy_passed`'s "no baseline, skip
        the check" sentinel -- a measured zero there bypasses the gate for every
        later candidate.

        Args:
            result: The baseline result dict, mutated in place.
            salvaged: The parsed eval dict from
                :meth:`_salvage_sibling_baseline_accuracy`.
            shared_state: The live SharedState, or ``None``.
            expected_handoff: Whether this read is the double-run design
                (see :func:`_is_double_run_accuracy_handoff`) rather than a
                recovery. The structured warning is for the recovery only;
                raising it on every healthy double-run baseline leaves the
                record claiming a fault the run never hit.

        Returns:
            float: The salvaged accuracy.
        """
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
            try:
                shared_state.baseline_accuracy = acc_val
            except Exception:  # noqa: BLE001 — salvage must never break baseline
                log.debug("baseline_executor: salvage could not set shared_state", exc_info=True)
        return acc_val

    def _salvage_sibling_baseline_accuracy(
        self,
        result: dict[str, Any],
        framework: str | None,
    ) -> dict[str, Any] | None:
        """Return a measured accuracy from a sibling baseline attempt, if any.

        A measured ``0.0`` is returned like any other score: it is evidence,
        not an absent measurement. The caller decides whether the value is
        usable; filtering zeros here would make a real quality failure
        indistinguishable from "the eval never ran".

        Scans the shared ``runs/baseline`` root (the parent of this attempt's
        ``output_dir``) so eval output written by any sibling attempt is seen.
        Discarded warmup rounds are excluded by :func:`parse_eval_results`.
        Best-effort: any error yields ``None`` (the caller then stops as before).

        Args:
            result: The final baseline result dict (carries ``output_dir``).
            framework: Framework name threaded into the eval parser.

        Returns:
            The parsed eval dict when a finite accuracy is found, else
            ``None``.
        """
        out = result.get("output_dir")
        if not out:
            return None
        runs_root = Path(out).parent  # .../runs/baseline
        if not runs_root.exists():
            return None
        try:
            from ._accuracy_gate import _finite_score, parse_eval_results

            eval_data = parse_eval_results(runs_root, framework=framework)
        except Exception:  # noqa: BLE001 — salvage must never break the stop path
            log.debug("baseline_executor: sibling-accuracy salvage scan failed", exc_info=True)
            return None
        if _finite_score(eval_data.get("accuracy")) is None:
            return None
        return eval_data

    async def _run_once(
        self,
        ctx: RunnerContext,
        *,
        force_disable_eval: bool = False,
        force_drop_moe_runner_backend: bool = False,
    ) -> dict[str, Any]:
        """Run the Magpie baseline benchmark and parse its result.

        Materializes the workload config, resolves the timeout (with cold-start
        detection), restarts the multi-node server when required, launches
        Magpie via ``run_with_session_kill``, harvests leaked artifacts, parses
        ``benchmark_report.json`` and the accuracy eval, and returns a result
        dict the Coordinator promotes into SharedState.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                (config / model / timeout knobs) and ``extra`` (workspace).
            force_disable_eval: When True, force ``RUN_EVAL=false`` into the
                materialized config (the eval-failure fallback path); also set
                by the ``disable_run_eval`` task param.
            force_drop_moe_runner_backend: When True, strip any inherited
                ``--moe-runner-backend`` and skip the AMD MoE injection (the
                one-shot fallback after the backend killed the server).

        Returns:
            dict[str, Any]: On success, a ``status="succeeded"`` dict with
                throughput / latency / accuracy measurements and artifact
                paths; on failure, a ``status="failed"`` dict with an
                ``error_class`` (``timeout``, ``subprocess_nonzero``,
                ``no_workspace``, ``no_report``, ``invalid_measurement`` ...).

        Raises:
            FileNotFoundError: If the resolved baseline config does not exist.
        """
        params = ctx.task.params or {}
        # Only a genuine ``baseline`` task may establish/overwrite the quality
        # reference; ``replay_warm_recipe`` reuses this executor but must compare
        # against the pure baseline reference rather than redefine it.
        is_genuine_baseline = _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params)
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")

        # One-shot cuda-graph eager fallback: a prior baseline armed
        # state.baseline_eager_fallback. Inject the framework-correct
        # disable-cuda-graph flag and consume the flag so it fires once. Resolve
        # framework FIRST: an unknown framework leaves the flag armed (not
        # consumed) for a later baseline instead of burning the one-shot.
        effective_extra_server_args = str(params.get("extra_server_args") or "")
        extra = getattr(ctx, "extra", None) or {}
        live_shared_state = extra.get("shared_state") or self.shared_state
        fw = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip()
        if not fw and self._eager_fallback_armed(live_shared_state):
            log.warning(
                "baseline_executor: eager fallback is armed but framework is "
                "unknown; leaving the one-shot armed (not consuming) so a "
                "later baseline with a known framework can apply it",
            )
        elif fw and self._consume_eager_fallback(live_shared_state):
            cg_flag = _disable_cuda_graph_flag(fw)
            effective_extra_server_args = _with_cuda_graph_disabled(
                effective_extra_server_args,
                fw,
            )
            log.warning(
                "baseline_executor: retrying with %s after a prior cuda-graph capture failure (framework=%s)",
                cg_flag,
                fw,
            )
        if force_drop_moe_runner_backend:
            effective_extra_server_args = _remove_moe_runner_backend_arg(effective_extra_server_args)
        if effective_extra_server_args or "extra_server_args" in params:
            # Keep the task envelope aligned with the materialized runtime so
            # Roofline fingerprints record one-shot eager fallback accurately.
            params["extra_server_args"] = effective_extra_server_args

        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Keep the InferenceX checkout Magpie ``cd``s into on stable local disk
        # so SGLang's relative-path cuda-graph dump survives a wekafs/NFS flap.
        # Relocate BEFORE materialize so the rendered ``benchmark.inferencex_path``,
        # the ProfileExecutor patch step, and Magpie all use the local mirror.
        # Kept task-local (not process-wide $INFERENCEX_PATH) to avoid races
        # between overlapping asyncio tasks.
        ix_env = os.environ.get("INFERENCEX_PATH", "").strip()
        effective_inferencex_path = _ensure_local_inferencex(ix_env, mirror_key=str(output_dir)) if ix_env else ""

        # Warm patches are prepared after config/runtime preflight, immediately
        # before the single final benchmark.
        # Explore/Framework Recipe patches target the framework checkout, not
        # the InferenceX benchmark harness. The Session's explicitly selected
        # root is the sole authority, matching Kernel Recipe replay.
        patch_target = _resolve_recipe_patch_target(params)
        patch_application: list[dict[str, str]] | dict[str, Any] = []
        applied_patches: list[dict[str, str]] = []
        _pre_patch_sha = ""

        timeout_sec = self._resolve_timeout(params)
        # Model path: unified resolver (params → $MODEL_PATH → SharedState), then
        # serving-path normalization (HL_MODEL_BASE / HF cache). If none, leave
        # the YAML's hardcoded `model:` for fixture-based tests.
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(live_shared_state, "model_path", "") or ""),
            for_serving=True,
        )
        # gpu_type: task.params > $GPU_TYPE (cli.py canonicalizes mi325x->mi300x).
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        # Orchestration-supplied script + result_dir overrides. Sanitization
        # turns a malformed override into ``error_class=bad_param``.
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
        # Accuracy eval (GSM8K) opt-out: ``--no-eval``, the ``disable_run_eval``
        # param and the eval-failure fallback force ``RUN_EVAL=false``. Candidates
        # template from this materialized YAML, so they inherit it.
        base_extra_envs = dict(params.get("extra_envs") or {})
        eval_disabled = self._eval_disabled(ctx)
        # The staged accuracy round is itself an eval, so ``--no-eval`` cancels it.
        defer_accuracy_until_after_measure = not eval_disabled and is_truthy(
            params.get("defer_accuracy_until_after_measure")
        )
        if force_disable_eval or is_truthy(params.get("disable_run_eval")) or eval_disabled:
            base_extra_envs["RUN_EVAL"] = "false"
        try:
            config_path = materialize_config_with_envs(
                config_path,
                output_dir,
                extra_server_args=effective_extra_server_args,
                extra_envs=base_extra_envs,
                remove_args=params.get("remove_args"),
                unset_envs=params.get("unset_envs"),
                args_mode=str(params.get("args_mode") or "append"),
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                inferencex_path=effective_inferencex_path,
                benchmark_script=override_script,
                establish_quality_ref=is_genuine_baseline,
                drop_moe_runner_backend=force_drop_moe_runner_backend,
                flydsl_source_dirs=is_truthy(params.get("flydsl_source_dirs")),
            )
        except FrameworkScriptMismatchError as exc:
            # Cross-framework script override: return a structured failure.
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        # Stash for the result so Coordinator can reuse it downstream.
        materialized_config_path = config_path
        # Apply runtime_override from params into the materialized YAML so the
        # revalidation baseline boots under the same framework runtime as the
        # KEEP'd candidate (PATH/PYTHONPATH/framework_bin etc.).
        _rt_from_params = params.get("runtime_override")
        if isinstance(_rt_from_params, dict) and _rt_from_params:
            try:
                import yaml as _yaml

                from ._grid_runner import apply_runtime_override

                _cfg_data = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                _cfg_bench = _cfg_data.setdefault("benchmark", {})
                _cfg_envs = _cfg_bench.setdefault("envs", {})
                apply_runtime_override(_cfg_envs, _rt_from_params)
                config_path.write_text(_yaml.safe_dump(_cfg_data), encoding="utf-8")
            except Exception:  # noqa: BLE001 — runtime overlay is best-effort
                log.debug("baseline_executor: runtime_override application failed", exc_info=True)
        # AgentX: deploy the aiperf client into InferenceX benchmarks/ and
        # capability-preflight aiperf before Magpie runs the materialized config.
        # Baseline/profile shell out here (not via _run_magpie), so without this the
        # materialize-time swap to aiperf_client.sh would point at a script that was
        # never deployed. No-op when HYPERLOOM_AGENTX is off.
        _agx_err = prepare_agentx_runtime(
            env=os.environ,
            inferencex_path=effective_inferencex_path,
            config_path=config_path,
        )
        if _agx_err:
            return {
                "status": "failed",
                "error_class": "agentx_preflight",
                "error": _agx_err,
                "output_dir": str(output_dir),
            }
        # Whether THIS run actually executes lm-eval, read back from the
        # materialized config the subprocess consumes -- the single source of
        # truth. ``materialize_config_with_envs`` folds RUN_EVAL from the base
        # YAML ``benchmark.envs``, ``reference_envs``, ``extra_envs`` (incl. the
        # eval-failure fallback / ``disable_run_eval`` force-off), and process
        # ``$RUN_EVAL`` (defaulting to "true"), so deriving from ``extra_envs``
        # alone would miss the other sources. Threaded into
        # ``_run_single_benchmark`` so accuracy is parsed ONLY when eval ran --
        # never salvaging a prior attempt's stale ``results*.json`` from the
        # reused per-round slot.
        hook_result = self._after_materialize_config(config_path, output_dir)
        if hook_result is not None:
            hook_result.setdefault("materialized_config", str(config_path))
            hook_result.setdefault("output_dir", str(output_dir))
            return hook_result

        # Cold-start "warmup artifact" guard: the freshly-booted server's first
        # benchmark window pays one-time cold costs that inflate later gains into
        # fictitious "improvements". Run TWICE against the same persistent server
        # via ``server_lifecycle`` reuse — round 1 pays cold costs, round 2
        # re-attaches to the hot server and is the clean baseline. Eligibility
        # (else single round): double-run requested, single-node, built-in
        # benchmark script, profiler off.
        lifecycle = self._resolve_lifecycle_params(materialized_config_path)
        double_run_requested = self._double_run_enabled(
            params=params,
            ctx_extra=extra,
        )
        double_run = double_run_requested and lifecycle["eligible"]
        if defer_accuracy_until_after_measure and double_run:
            # Only the lifecycle path can reuse the hot server for a staged
            # accuracy round. A single-round fallback must retain the original
            # eval contract so a throughput winner still carries accuracy.
            _set_materialized_run_eval(
                materialized_config_path,
                enabled=False,
            )
        run_eval_disabled = materialized_run_eval_disabled(materialized_config_path)

        # Asked before the lease, because a round that will not be run should not
        # hold a GPU while being refused.
        ignitable, ignition_evidence = self._round_affordable_before_ignition(
            double_run=double_run,
            ctx_extra=extra,
        )
        if not ignitable:
            stopped_result = _stopped_round_result(
                STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
                round_label="baseline round",
                returncode=None,
                runtime_sec=0.0,
                output_dir=output_dir,
                capture_meta={
                    "materialized_config": str(materialized_config_path),
                    "run_eval_disabled": bool(run_eval_disabled),
                },
                started=False,
            )
            stopped_result["budget_shortfall"] = ignition_evidence
            if ignition_evidence.get("one_more_measurement_sec"):
                log.warning(
                    "baseline_executor: this round (%.0fs) and one variant to read "
                    "against it (%.0fs) need %.0fs, and %.0fs is left (bound=%s), so "
                    "nothing is booted. A baseline no variant can follow is a "
                    "denominator with no numerator; the anchor this session already "
                    "measured stands.",
                    ignition_evidence.get("round_sec", 0.0),
                    ignition_evidence.get("one_more_measurement_sec", 0.0),
                    ignition_evidence.get("expected_cost_sec", 0.0),
                    ignition_evidence.get("affordable_sec", 0.0),
                    ignition_evidence.get("bound", ""),
                )
            else:
                log.warning(
                    "baseline_executor: this round needs %.0fs and only %.0fs is left "
                    "(bound=%s), so nothing is booted. The anchor this session already "
                    "measured stands.",
                    ignition_evidence.get("expected_cost_sec", 0.0),
                    ignition_evidence.get("affordable_sec", 0.0),
                    ignition_evidence.get("bound", ""),
                )
            return stopped_result

        before_apply_sha = _git_head_sha(patch_target) if patch_target else ""

        def _persist_recipe_snapshot(manifest: dict[str, Any]) -> bool:
            if live_shared_state is None:
                return True
            pending = dict(getattr(live_shared_state, "warm_replay_pending", {}) or {})
            pending.update(
                {
                    "status": "preparing_required_recipe",
                    "recipe_patch_target": patch_target,
                    "recipe_patch_pre_sha": before_apply_sha,
                    "recipe_patch_snapshot_manifest": manifest,
                }
            )
            live_shared_state.warm_replay_pending = pending
            try:
                live_shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.warning(
                    "combined warm replay recipe snapshot persist failed",
                    exc_info=True,
                )
                return False
            return True

        if params.get("patches") and not patch_target:
            patch_application = {
                "status": "failed",
                "failure": "missing_target_repo",
                "patches": [],
                "applied": [],
            }
        else:
            patch_application = _apply_warm_patches(
                params,
                patch_target,
                output_dir,
                before_mutation=_persist_recipe_snapshot,
            )
        if isinstance(patch_application, dict):
            applied_patches = list(patch_application.get("applied") or [])
            _pre_patch_sha = str(patch_application.get("pre_sha") or "")
            if patch_application.get("status") == "failed":
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
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "combined warm replay cleanup persist failed",
                            exc_info=True,
                        )
                return {
                    "status": ("required_patch_failed" if rollback_result["ok"] else "required_patch_rollback_failed"),
                    "error_class": (
                        "required_patch_failed" if rollback_result["ok"] else "warm_replay_rollback_failed"
                    ),
                    "error": str(patch_application.get("failure") or "required recipe timeline patch failed"),
                    "required_patch_failure": patch_application,
                    "failed_patch_ref": patch_application.get("failed_ref"),
                    "warm_kernel_rolled_back": bool(kernel_rollback.get("ok")),
                    "warm_replay_rollback": rollback_result,
                    "workspace": str(output_dir),
                }
            if live_shared_state is not None:
                pending = dict(getattr(live_shared_state, "warm_replay_pending", {}) or {})
                pending.update(
                    {
                        "status": "benchmarking",
                        "recipe_patch_target": patch_target,
                        "recipe_patch_pre_sha": _pre_patch_sha,
                        "recipe_patch_snapshot_manifest": patch_application.get("snapshot_manifest"),
                        "recipe_patch_statuses": list(patch_application.get("patches") or []),
                    }
                )
                live_shared_state.warm_replay_pending = pending
                try:
                    live_shared_state.save(self.session_dir)
                except Exception:  # noqa: BLE001
                    log.debug(
                        "combined warm replay pending persist failed",
                        exc_info=True,
                    )
        else:
            applied_patches = patch_application
            _pre_patch_sha = before_apply_sha
        if applied_patches:
            log.info(
                "baseline_executor: prepared %d warm-replay code patches: %s",
                len(applied_patches),
                [p["patch_file"] for p in applied_patches],
            )
        # Ray-managed GPU execution (§12 T1): one held Ray lease (``num_gpus=TP``)
        # spans this baseline's benchmark rounds — a double-run's warmup +
        # measure reuse one persistent server, so both must run under the same
        # lease. ``None`` on the local path (multi-node / RAY_EXEC off / tests)
        # keeps the legacy behaviour. The lease is closed on every exit below.
        from ._grid_runner import _num_gpus_for_config
        from ._ray_serving import maybe_serving_lease

        bench_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(materialized_config_path))

        common = {
            "timeout_sec": timeout_sec,
            "override_result_dir": override_result_dir,
            "resolved_model": resolved_model,
            "materialized_config_path": materialized_config_path,
            "inferencex_path": effective_inferencex_path,
            "effective_extra_server_args": effective_extra_server_args,
            "params": params,
            "ctx": ctx,
            "run_eval_disabled": run_eval_disabled,
            "serving_lease": bench_lease,
        }

        if not double_run:
            if double_run_requested and not lifecycle["eligible"]:
                log.info(
                    "baseline_executor: cold-start double-run not eligible (%s); running single round.",
                    lifecycle["reason"],
                )
            try:
                result = await self._run_reported_round(
                    label="single",
                    config_path=config_path,
                    output_dir=output_dir,
                    **common,
                )
                if applied_patches:
                    result["warm_patches_applied"] = list(applied_patches)
                if isinstance(patch_application, dict):
                    result["warm_patch_result"] = patch_application
                    result["warm_patch_pre_sha"] = _pre_patch_sha
                    result["warm_patch_target"] = patch_target
                    result["warm_patch_snapshot_manifest"] = patch_application.get("snapshot_manifest")
                    result["warm_patch_canonical_target"] = patch_target
                    result["warm_kernel_apply_results"] = list(params.get("warm_kernel_apply_results") or [])
                return result
            finally:
                # A required timeline's tree is promoted by prelude after this
                # returns, so it must stay patched; reverting here handed prelude
                # a clean tree and silently lost the replay.
                if (
                    applied_patches
                    and not isinstance(patch_application, dict)
                    and (_pre_patch_sha or params.get("_warm_patch_nogit_backups"))
                ):
                    _revert_warm_patch_state(
                        patch_target,
                        pre_sha=_pre_patch_sha,
                        snapshot_manifest=params.get("_warm_patch_snapshot_manifest"),
                        nogit_backups=list(params.get("_warm_patch_nogit_backups") or []),
                    )
                if bench_lease is not None:
                    bench_lease.close()

        framework = lifecycle["framework"]
        port = lifecycle["port"]
        # pid_dir is shared across both rounds so round 2 discovers round 1's
        # server; task root keeps it per-task isolated.
        pid_dir = output_dir
        try:
            # Deep-clean zombie listeners + stale pid/meta BEFORE round 1 boots.
            # Runs once here so round 1's server survives for round 2's re-attach.
            self._pre_start_cleanup(
                pid_dir=pid_dir,
                framework=framework,
                port=port,
            )
            # Round 1 (warmup): boot + run, leave running so round 2 can
            # re-attach. Throughput discarded (cold-contaminated).
            warmup_dir = output_dir / "warmup_round"
            # A replayed KB config is promoted onto ``current_best`` and becomes
            # the reference every later measurement in the session is taken
            # against, so it may not be adopted on throughput alone. The warmup
            # round is the only round that evaluates, so forcing it here is what
            # gives the promotion gate a score to judge; inheriting a contract
            # with RUN_EVAL off would leave the gate with no evidence. The
            # staged-accuracy lane owns its own eval schedule and is left alone.
            # ``force_disable_eval`` marks the salvage retry taken when the eval
            # is itself what aborted the run; forcing it back on there would
            # reproduce the failure and lose the throughput baseline too.
            # ``--no-eval`` is the operator saying no eval runs this session, and
            # forcing one here would spend the time it was passed to save while
            # silently overriding that. A replay is then promoted unjudged, which
            # is the trade the flag already makes everywhere else.
            force_warmup_eval = (
                str(getattr(ctx.task, "kind", "") or "") == "replay_warm_recipe"
                and not defer_accuracy_until_after_measure
                and not force_disable_eval
                and not eval_disabled
            )
            warmup_cfg = self._write_lifecycle_config(
                materialized_config_path,
                warmup_dir,
                cleanup=False,
                pid_dir=pid_dir,
                port=port,
                run_eval=True if force_warmup_eval else None,
            )
            log.info(
                "baseline_executor: cold-start guard — warmup round (discarded, boots persistent server) in %s",
                warmup_dir,
            )
            # The warmup runs under the round's own cap, which the session clamp
            # leaves sitting past the session deadline so the watchdog reaches it
            # first and a budget kill is recorded as one. Whether the measured
            # round can follow is asked after this pass, priced with what it
            # actually cost rather than a prediction of what it would.
            warmup_result = await self._run_reported_round(
                label="warmup",
                config_path=warmup_cfg,
                output_dir=warmup_dir,
                **common,
            )
            if warmup_result.get("status") != "succeeded":
                # Warmup failure almost certainly recurs, so skip the
                # measured round.
                warmup_result.setdefault("nonfatal_warnings", [])
                warmup_result["nonfatal_warnings"].append(
                    "baseline_warmup_round_failed",
                )
                log.warning(
                    "baseline_executor: warmup round failed (error_class=%s); skipping measured round",
                    warmup_result.get("error_class"),
                )
                if applied_patches:
                    warmup_result["warm_patches_applied"] = list(applied_patches)
                if isinstance(patch_application, dict):
                    warmup_result["warm_patch_result"] = patch_application
                    warmup_result["warm_patch_pre_sha"] = _pre_patch_sha
                    warmup_result["warm_patch_target"] = patch_target
                    warmup_result["warm_patch_snapshot_manifest"] = patch_application.get("snapshot_manifest")
                    warmup_result["warm_patch_canonical_target"] = patch_target
                    warmup_result["warm_kernel_apply_results"] = list(params.get("warm_kernel_apply_results") or [])
                return warmup_result
            warmup_tput = warmup_result.get("output_throughput")
            warmup_runtime = warmup_result.get("subprocess_runtime_sec")
            warmup_post_ready = warmup_result.get("post_ready_runtime_sec")
            await report_progress(
                unit="baseline_round",
                label="warmup",
                index=1,
                total=2,
                status="succeeded",
                output_throughput=warmup_tput,
                runtime_sec=warmup_runtime,
            )

            if not defer_accuracy_until_after_measure:
                affordable, gate_evidence = self._measure_round_affordable(
                    warmup_runtime_sec=warmup_runtime,
                    warmup_post_ready_sec=warmup_post_ready,
                    ctx_extra=extra,
                )
                if not affordable:
                    if gate_evidence.get("one_more_measurement_sec"):
                        why = "a hot pass (%.0fs) and one variant to read against it (%.0fs) need %.0fs" % (
                            gate_evidence.get("measure_round_sec", 0.0),
                            gate_evidence.get("one_more_measurement_sec", 0.0),
                            gate_evidence.get("expected_cost_sec", 0.0),
                        )
                    else:
                        why = "a hot pass needs %.0fs" % (gate_evidence.get("expected_cost_sec", 0.0),)
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
                        gate_evidence.get("affordable_sec", 0.0),
                        gate_evidence.get("bound", ""),
                    )
                    return _cold_anchor_from_warmup(warmup_result, dropped=gate_evidence)

            # Round 2 (measured): re-attach to the hot server (client only).
            # Warm re-attach is intentional — all comparison points (baseline,
            # explore decision and their grading anchor) are
            # measured with a warm prefix cache, keeping them mutually
            # comparable. Carryover is config-dependent (tracks KV-block
            # capacity) and is not a uniform offset.
            #
            # No accuracy eval: ordinary baselines measured it in round 1;
            # staged kernel integration defers it until after this gate.
            measure_dir = output_dir / "measure_round"
            measure_cfg = self._write_lifecycle_config(
                materialized_config_path,
                measure_dir,
                cleanup=not defer_accuracy_until_after_measure,
                pid_dir=pid_dir,
                port=port,
                run_eval=False,
            )
            log.info(
                "baseline_executor: cold-start guard — measured baseline "
                "round in %s (warmup tput=%.1f tok/s discarded, reusing "
                "hot server)",
                measure_dir,
                warmup_tput or 0.0,
            )
            result = await self._run_reported_round(
                label="measure",
                config_path=measure_cfg,
                output_dir=measure_dir,
                **common,
            )
            if applied_patches:
                result["warm_patches_applied"] = list(applied_patches)
            if isinstance(patch_application, dict):
                result["warm_patch_result"] = patch_application
                result["warm_patch_pre_sha"] = _pre_patch_sha
                result["warm_patch_target"] = patch_target
                result["warm_patch_snapshot_manifest"] = patch_application.get("snapshot_manifest")
                result["warm_patch_canonical_target"] = patch_target
                result["warm_kernel_apply_results"] = list(params.get("warm_kernel_apply_results") or [])
            if result.get("status") != "succeeded" and result.get("error_class") == SESSION_TIME_EXHAUSTED_CLASS:
                # The gate before this pass admitted it and the run's clock took
                # it anyway -- the pass overran what it was priced at. Reporting
                # the round as failed would throw away the warmup's figure too,
                # leaving the session with nothing from GPU time it has already
                # spent, so the warmup is kept and marked exactly as a refusal
                # keeps it. Only a reap is handled this way: a pass that failed
                # for a reason of its own is a failure worth surfacing, and the
                # warmup having succeeded does not make it comparable.
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
                result.setdefault("nonfatal_warnings", [])
                result["nonfatal_warnings"].append(
                    "baseline_double_run_discarded_first",
                )
                result["warmup_round_tput"] = warmup_tput
                self._record_baseline_convergence(result, warmup_tput)
                # The Coordinator promotes ``subprocess_runtime_sec`` into the
                # explore soft-kill anchor. Explore variants restart the server,
                # so report round 1's full boot+client wall-clock; round 2's
                # client-only time stays under a separate key.
                if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
                    result["measure_round_runtime_sec"] = result.get(
                        "subprocess_runtime_sec",
                    )
                    result["subprocess_runtime_sec"] = round(
                        float(warmup_runtime),
                        2,
                    )
                    # The split belongs to round 1 for the same reason its
                    # wall-clock does: round 2 re-attached, so it has no boot to
                    # separate and its own reading says nothing about what booting
                    # this workload costs. Assigned with that wall-clock and not
                    # beside it, so the total and the part of it reported here can
                    # never come from different rounds -- their difference is
                    # published as this workload's boot.
                    result["post_ready_runtime_sec"] = warmup_result.get("post_ready_runtime_sec")
                _hot = result.get("output_throughput") or 0.0
                _cold = warmup_tput or 0.0
                log.info(
                    "baseline_executor: cold-start guard — measured "
                    "baseline=%.1f tok/s (warmup=%.1f tok/s discarded; "
                    "artifact would have been +%.0f%%)",
                    _hot,
                    _cold,
                    ((_hot / _cold - 1.0) * 100.0) if _cold > 0 else 0.0,
                )
                if defer_accuracy_until_after_measure:
                    try:
                        min_tput = float(
                            params.get(
                                "post_measure_accuracy_min_tput",
                                0.0,
                            )
                            or 0.0
                        )
                    except (TypeError, ValueError):
                        min_tput = 0.0
                    if float(_hot or 0.0) >= min_tput:
                        accuracy_dir = output_dir / "accuracy_round"
                        accuracy_cfg = self._write_lifecycle_config(
                            materialized_config_path,
                            accuracy_dir,
                            cleanup=True,
                            pid_dir=pid_dir,
                            port=port,
                            run_eval=True,
                        )
                        try:
                            accuracy_timeout_sec = int(params.get("accuracy_timeout_sec") or timeout_sec)
                        except (TypeError, ValueError):
                            accuracy_timeout_sec = timeout_sec
                        accuracy_result = await self._run_reported_round(
                            label="accuracy",
                            config_path=accuracy_cfg,
                            output_dir=accuracy_dir,
                            **{
                                **common,
                                "timeout_sec": max(
                                    1,
                                    accuracy_timeout_sec,
                                ),
                                "run_eval_disabled": False,
                            },
                        )
                        result["accuracy_stage"] = {
                            "status": accuracy_result.get("status"),
                            "error_class": accuracy_result.get("error_class"),
                            "workspace": accuracy_result.get("workspace"),
                        }
                        if accuracy_result.get("status") == "succeeded":
                            for key in (
                                "accuracy",
                                "accuracy_task",
                                "accuracy_metric",
                                "accuracy_source",
                            ):
                                if accuracy_result.get(key) is not None:
                                    result[key] = accuracy_result[key]
                        else:
                            result.setdefault("nonfatal_warnings", [])
                            result["nonfatal_warnings"].append("post_measure_accuracy_failed")
                    else:
                        result["accuracy_stage"] = {
                            "status": "skipped",
                            "reason": "throughput_below_threshold",
                            "minimum_tput": min_tput,
                            "observed_tput": float(_hot or 0.0),
                        }
            return result
        finally:
            # Defensive teardown so no persistent server leaks. Idempotent.
            # Reap the server BEFORE releasing the Ray lease so no GPU process
            # outlives it (§4.2).
            self._teardown_lifecycle_server(
                pid_dir=pid_dir,
                framework=framework,
                port=port,
            )
            # Revert warm-replay patches to prevent state leakage into
            # subsequent tasks that reuse the same InferenceX checkout. A
            # required timeline is exempt: prelude promotes that tree after this
            # returns and needs it still patched.
            if (
                applied_patches
                and not isinstance(patch_application, dict)
                and (_pre_patch_sha or params.get("_warm_patch_nogit_backups"))
            ):
                _revert_warm_patch_state(
                    patch_target,
                    pre_sha=_pre_patch_sha,
                    snapshot_manifest=params.get("_warm_patch_snapshot_manifest"),
                    nogit_backups=list(params.get("_warm_patch_nogit_backups") or []),
                )
            if bench_lease is not None:
                bench_lease.close()

    def _round_affordable_before_ignition(
        self,
        *,
        double_run: bool,
        ctx_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a whole round *and a use for it*, before anything boots.

        The companion to :meth:`_measure_round_affordable`, and the two divide
        the session's baselines between them. A session's first round is not
        asked this question, because there is nothing to ask it with: the
        measured runtimes are written only once an anchor lands, so a gate here
        would either refuse every first baseline or wave every one through. That
        round runs, and whether its second pass may follow is settled afterwards
        against what the first actually cost.

        Every later round is asked, and a resumed session's first round is the
        case that most needs it: it holds the earlier leg's measurements, so a
        round that cannot lead anywhere can be refused before a single second of
        GPU time is spent on it, and the refusal names a number an operator can
        act on.

        What is required in PRELUDE is the round *plus one further measured
        variant*, not the round alone. A baseline is not a result there; it is the
        denominator later results are read against, and one that nothing is ever
        compared to is wall-clock spent on a number no one uses. A variant's own
        measurement is a result, which is why the variant gates ask only whether
        the variant itself fits -- and why a re-baseline in a later phase is asked
        the same narrower question by :func:`_a_use_must_follow_the_round`.

        The round is priced by
        :func:`~...phases.machine_state.baseline_round_cost_sec`, which the phase
        machine also reads to decide whether a session stopped for this reason may
        try again on a fresh clock. Two definitions would let the executor refuse
        rounds the phase machine had just decided were affordable.

        Args:
            double_run: Whether this round will run both passes.
            ctx_extra: The runner context extras carrying ``shared_state``.

        Returns:
            tuple[bool, dict[str, Any]]: ``(affordable, evidence)``.
        """
        state = (ctx_extra or {}).get("shared_state") or self.shared_state
        return self._round_affordable(
            state,
            round_sec=_phase_state.baseline_round_cost_sec(state, double_run=double_run),
        )

    @staticmethod
    def _round_affordable(state: Any, *, round_sec: float | None) -> tuple[bool, dict[str, Any]]:
        """Whether the budget holds a round costing ``round_sec`` and a use for it.

        The shared half of every before-ignition gate, so the two round shapes a
        baseline has -- the single-node cold-then-hot double run and the
        multi-node pair of client passes -- differ only in what they cost, never
        in what they are asked. Each supplies its own price and this decides.

        Args:
            state: The session ``SharedState``, or ``None``.
            round_sec: What this round is expected to cost, or ``None`` when the
                session has measured nothing to price it from.

        Returns:
            tuple[bool, dict[str, Any]]: ``(affordable, evidence)``.
        """
        cold_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
        if cold_sec is None or round_sec is None:
            return True, {"reason": "no_measured_round_to_predict_from"}
        use_sec = 0.0
        if _a_use_must_follow_the_round(state):
            # Without the split, a variant is priced at a whole cold round, which
            # is what it is: a boot and a benchmark that pays the compile. The
            # same fallback the phase machine uses, so the two agree on when a
            # stopped session may try again.
            #
            # Known gap: a multi-node variant runs two client passes, not one
            # (``_grid_runner`` reserves for it as ``x (1 + _mn_warmup_rounds)``),
            # so one pass is left unreserved here. It needs a multi-node round to
            # reach PRELUDE with an earlier one already measured, which takes an
            # enablement round holding the phase open past an anchor that would
            # otherwise finish it -- narrow enough not to be worth teaching the
            # phase machine's pricing what shape the cluster is.
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
        """Whether the budget covers the measured round *and a use for it*.

        Asked after the warmup rather than before it, and priced from what that
        pass just measured rather than from a prediction. This is the gate every
        round faces, including the first, which is the one
        :meth:`_round_affordable_before_ignition` cannot judge.

        The measured round re-attaches to the server the warmup left running, so
        it costs a benchmark and no boot. A hot pass this session already measured
        prices it best, being exactly that; before one has ever run, the warmup's
        post-ready segment stands in -- the same pass, with its boot taken off. The
        segment over-predicts, since it also paid the first request's kernel
        compile, but it is far tighter than the warmup's whole wall-clock, which
        prices a client-only pass as though it booted a server.

        Reading the session's hot figure first is also what keeps this gate and
        :meth:`_round_affordable_before_ignition` on one ruler. Both price the
        same second pass, so a round admitted before ignition would otherwise meet
        a stricter question here and be refused for certain, having spent a whole
        cold pass to find out.

        In PRELUDE, covering the pass is not enough to justify running it. A hot
        baseline is an input there, not a result: it is the denominator later
        variants are read against, and it is what anchors their overtime kill.
        Neither buys anything if no variant can follow, so what is required is the
        pass plus one further measured variant -- a boot and a benchmark, because
        a variant's config differs in the very knobs that decide how a server
        comes up and it cannot re-attach to anyone else's. A re-baseline in a
        later phase is its own deliverable and is asked only to cover its pass;
        :func:`_a_use_must_follow_the_round` draws that line.

        A round that fails here has still produced a number. It is the cold one
        the double run exists to discard, so the caller keeps it as the anchor and
        marks it -- the GPU time is spent either way, and a marked cold anchor
        beats no anchor.

        Args:
            warmup_runtime_sec: Wall-clock the warmup round took.
            warmup_post_ready_sec: The part of it that ran after the server was
                ready. Only a stamp that could not be written leaves this unset,
                since this path runs a server by definition -- the workloads with
                no ready boundary to record never reach a double run at all -- so
                it is a defect rather than a shape, and the gate waves the round
                through rather than guess. Guessing high is what a gate before
                ignition may safely do; here a refusal ends the session, and a
                session ended by a missing timestamp is the worse error.
            ctx_extra: The runner context extras carrying ``shared_state``.

        Returns:
            tuple[bool, dict[str, Any]]: ``(affordable, evidence)``.
        """
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

    def _double_run_enabled(
        self,
        *,
        params: dict[str, Any] | None = None,
        ctx_extra: dict[str, Any] | None = None,
    ) -> bool:
        """Whether baseline double-run is enabled.

        Public CLI/env controls are intentionally unsupported. The session
        default is on so the warm decision compares hot candidates against a
        hot baseline. Internal callers may pass
        ``task.params["baseline_double_run"]`` for focused tests/debug runs, or
        set the session state directly.

        Returns:
            ``True`` unless task params or session state explicitly opt out.
        """
        params = params or {}
        if "baseline_double_run" in params:
            return is_truthy(params.get("baseline_double_run"))

        extra = ctx_extra or {}
        state = extra.get("shared_state") or self.shared_state
        if state is not None:
            return bool(getattr(state, "baseline_double_run", False))

        try:
            from ...state.shared_state import SharedState

            session_dir = Path(str(extra.get("session_dir") or self.session_dir))
            state = SharedState.load_or_init(session_dir)
            return bool(getattr(state, "baseline_double_run", False))
        except Exception:  # noqa: BLE001 - keep baseline fallback double-run.
            log.debug(
                "baseline_executor: could not resolve baseline_double_run from session state",
                exc_info=True,
            )
            return True

    def _resolve_lifecycle_params(
        self,
        materialized_config_path: Path,
    ) -> dict[str, Any]:
        """Inspect the materialized YAML for server_lifecycle eligibility.

        Args:
            materialized_config_path: The materialized Magpie YAML config path.

        Returns:
            Lifecycle params including eligibility, framework, port and the
            reason a run is ineligible.
        """
        return _lifecycle.resolve_lifecycle_params(materialized_config_path)

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
        """Render a per-round YAML injecting ``benchmark.server_lifecycle``.

        Both rounds share ``pid_dir`` + ``port`` so round 2 re-attaches;
        ``cleanup`` and ``run_eval`` can differ between rounds.

        Args:
            base_config_path: Source materialized YAML to clone and patch.
            dest_dir: Directory the per-round YAML is written into.
            cleanup: Whether the server should be torn down after the round.
            pid_dir: Shared pid/metadata directory keying the persistent
                server across both rounds.
            port: Server port shared across both rounds.
            run_eval: Explicitly forces ``RUN_EVAL`` when set; ``None`` keeps
                the materialized benchmark contract unchanged.

        Returns:
            Path to the written per-round lifecycle YAML.
        """
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

    @staticmethod
    def _port_healthy(port: int, timeout: float = 3.0) -> bool:
        """Return True when localhost:{port}/health responds HTTP 200.

        Args:
            port: Local server port to probe.
            timeout: Per-request timeout in seconds.

        Returns:
            ``True`` when the health endpoint responds HTTP 200, else
            ``False``.
        """
        import urllib.request

        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=timeout,
            )  # nosec B310 - fixed loopback health check.
            return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _pre_start_cleanup(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Best-effort startup pre-clean for the double-run path.

        Only acts when there is concrete evidence of a zombie: the reuse
        port responds to /health but the matching metadata file is absent
        (the exact "Reuse metadata mismatch" trigger). In that case it
        calls _kill_stale_servers() to reap the orphan listener. Stale
        pid/json files are always unlinked (without sending signals to
        potentially-recycled PIDs). Never raises.

        Args:
            pid_dir: Directory holding the server pid/metadata files.
            framework: Framework name used to build the server tag.
            port: Server port used to build the server tag.
        """
        base = Path(pid_dir)
        tag = f"{framework}_{port}"
        pid_file = base / f"{tag}.pid"
        meta_file = base / f"{tag}.json"
        meta_exists = meta_file.exists()
        try:
            port_healthy = self._port_healthy(port)
        except Exception as exc:  # noqa: BLE001 — best-effort pre-clean
            log.warning(
                "baseline_executor: pre-start port probe failed (%s); proceeding.",
                exc,
            )
            port_healthy = False
        if meta_exists and port_healthy:
            # A healthy reuse target with metadata is not a zombie; keep the
            # files so Magpie can reattach.
            return
        # Unlink stale metadata/pid files only (no signal to possibly-recycled
        # PIDs).
        for p in (pid_file, meta_file):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        # Only deep-clean when the port is occupied by a zombie (healthy
        # endpoint, no metadata), to avoid killing unrelated servers.
        try:
            if not meta_exists and port_healthy:
                _kill_stale_servers()
        except Exception as exc:  # noqa: BLE001 — best-effort pre-clean
            log.warning(
                "baseline_executor: pre-start _kill_stale_servers failed (%s); proceeding.",
                exc,
            )

    def _teardown_lifecycle_server(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Best-effort teardown of a persistent server left by the
        double-run rounds. Idempotent and never raises (safe in finally).

        Args:
            pid_dir: Directory holding the server pid/metadata files.
            framework: Framework name used to build the server tag.
            port: Server port used to build the server tag.
        """
        _lifecycle.teardown_lifecycle_server(
            pid_dir=pid_dir,
            framework=framework,
            port=port,
        )

    async def _run_reported_round(
        self,
        *,
        label: str,
        config_path: Path,
        output_dir: Path,
        **common: Any,
    ) -> dict[str, Any]:
        """Announce a benchmark round before it blocks, then run it.

        Reported on entry, not on completion: a round can boot a server, warm
        JIT and bench for the better part of an hour, and one that never
        returns is exactly the case the heartbeat has to be able to show.

        Args:
            label (str): Round name carried on the progress note
                (``"single"``, ``"warmup"``, ``"measure"``, ``"accuracy"``).
            config_path (Path): The materialized Magpie YAML for this round.
            output_dir (Path): The per-round workspace slot.
            **common (Any): Remaining :meth:`_run_single_benchmark` arguments.

        Returns:
            dict[str, Any]: The round's benchmark result, unchanged.
        """
        await report_progress(unit="baseline_round", label=label, status="started")
        return await self._run_single_benchmark(
            config_path=config_path,
            output_dir=output_dir,
            **common,
        )

    async def _mn_warmup_pass(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
        output_dir: Path,
        framework: str,
        timeout_sec: int,
        session_deadline_sec: float | None,
        capture_meta: dict[str, Any],
        round_warnings: list[str],
        ctx_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run the discarded multi-node client warmup against the restarted server.

        The pass exists because the restart just above it left a cold server and
        this is the only thing that drives it before the measured pass. So the
        two are one round: skipping the warmup does not save the round's cost,
        it moves the round's measurement onto a cold server and anchors the
        session's every later gain on it.

        Both passes run under the round's own cap, which the session clamp leaves
        past the session deadline so a budget kill arrives as the watchdog's
        sentinel rather than as this pass timing out. Nothing is held back for
        the measured pass: holding back moves the cap in front of the deadline
        and puts the round back in reach of its own timeout.

        Args:
            cmd: The measured pass's command, re-pointed at the warmup slot.
            env: The measured pass's environment, re-pointed the same way.
            output_dir: The round's workspace; the warmup runs in a slot under it.
            framework: Framework name, for the watchdog's server log.
            timeout_sec: The round's cap after the session clamp.
            session_deadline_sec: Monotonic-clock session deadline, or ``None``.
            capture_meta: Config/eval-contract facts every failure result carries.
            round_warnings: Collected onto the round's result, for a warmup that
                did not run and did not end the round.
            ctx_extra: The runner context extras carrying ``shared_state``, read
                to price the pair of passes against what is left of the budget.

        Returns:
            dict[str, Any] | None: The round's result when the round is over,
                else ``None`` to go on to the measured pass.
        """
        # A multi-node round is two client passes of the same shape against a
        # server the round did not boot, and the session's measured figure is one
        # of them -- the round's wall-clock is taken after this pass -- so the pair
        # costs twice it. Asked before the warmup because that is the pass whose
        # number nothing may use: spending it and then meeting the deadline in the
        # measured pass leaves the round with no anchor and the session with the
        # GPU time gone.
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
        # The measurement is discarded, but the returncode is not: this pass is a
        # full benchmark round, so a stop here ends the baseline round. Going on
        # to the measured pass would spend a second round of GPU time the run has
        # already been told to stop spending.
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
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result.

        Single-round core extracted from ``__call__`` so the cold-start
        guard can invoke it twice. ``output_dir`` is the per-round slot.

        Args:
            config_path: The materialized Magpie YAML config for this round.
            output_dir: The per-round workspace slot.
            timeout_sec: Subprocess timeout in seconds.
            override_result_dir: Optional ``$RESULT_DIR`` override for the
                benchmark wrapper.
            resolved_model: Resolved model path for the run.
            materialized_config_path: The canonical materialized YAML, echoed
                into the result for downstream reuse.
            inferencex_path: Task-local InferenceX checkout path pinned via
                ``MAGPIE_INFERENCEX_PATH``.
            effective_extra_server_args: Extra server args passed to the
                multi-node restart helper.
            params: Task params for this launch.
            ctx: Runner context carrying ``extra`` (e.g. multi-node round
                state).
            run_eval_disabled: When True, this run did not execute the serving
                lm-eval (RUN_EVAL forced off via the eval-failure fallback,
                ``disable_run_eval``, or a present-and-falsey ``extra_envs``
                RUN_EVAL), so the serving GSM8K parse is skipped to avoid reading
                a prior attempt's stale ``results*.json`` from the reused slot.
                Scriptable frameworks are unaffected: RUN_EVAL does not gate
                their per-run image ``quality_gate``, which is still parsed.
            serving_lease: When set (Ray-managed GPU execution, §12 T1), the
                Magpie subprocess runs inside the lease's actor — which holds
                ``num_gpus`` across this run's rounds (double-run warmup +
                measure share one lease) — instead of a local subprocess. Ray
                owns ``*_VISIBLE_DEVICES``, so the YAML device list is stripped
                first (T2). ``None`` keeps the local ``run_with_session_kill``
                path unchanged.

        Returns:
            A result dict: ``status="succeeded"`` with measurements on
            success, or ``status="failed"`` with an ``error_class`` on
            failure.
        """
        cmd = build_benchmark_command(
            python_exe=self.magpie_python,
            config_path=config_path,
            output_dir=output_dir,
        )
        env = scrub_benchmark_process_env(os.environ.copy())
        # Put the venv first in PATH so the benchmark script's `python3`
        # resolves to one with torch+rocm (defense in depth vs Magpie YAML).
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        # Pin Magpie's InferenceX resolution to the same per-task
        # checkout rendered into benchmark.inferencex_path and patched by
        # ProfileExecutor. Do not re-read process env here; the task-local
        # explicit value avoids cross-task races on $INFERENCEX_PATH.
        if inferencex_path:
            env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Always-on ``$RESULT_DIR`` default for scripts that respect it; scripts
        # that ignore it are caught by the salvage pass.
        result_dir = _resolve_result_dir(output_dir, override_result_dir)
        env["RESULT_DIR"] = str(result_dir)
        # Config/eval-contract facts echoed onto every failure result so an
        # eval-rooted failure can be fingerprinted and re-run by enablement.
        capture_meta = {
            "materialized_config": str(materialized_config_path),
            "result_dir": str(result_dir),
            "run_eval_disabled": bool(run_eval_disabled),
        }
        # InferenceX ``run_lm_eval`` cleans ``$EVAL_RESULT_DIR`` after processing
        # lm-eval output. Keep it under the task workspace but separate from
        # Magpie's ``benchmark_*`` traces in ``$RESULT_DIR``.
        env["EVAL_RESULT_DIR"] = str(result_dir / "eval_output")
        # Pin SERVER_LOG / GPU_METRICS_CSV per-task so wrappers write into the
        # task workspace; ``harvest_leaked_artifacts`` is the defense-in-depth net.
        env["SERVER_LOG"] = str(output_dir / "server.log")
        env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

        # The materialized YAML is authoritative for the framework: params and
        # $FRAMEWORK are both optional on a baseline task.
        framework = (
            _config_framework(materialized_config_path)
            or str(params.get("framework") or "").strip().lower()
            or os.environ.get("FRAMEWORK", "").strip().lower()
        )
        watchdog_server_log = _watchdog_server_log_path(output_dir, framework)

        # Multi-node (--nodes >= 2): inject MAGPIE_RUN_PHASE=client +
        # BENCHMARK_BASE_URL so Magpie skips its server launch and targets
        # the RayJob head. No-op ({}) in single-node.
        from ._multi_node_env import magpie_remote_env

        env.update(magpie_remote_env())

        # Multi-node only: restart sglang/vllm per round for a fresh server.
        # No-op in single-node. Profile rounds set
        # ctx.extra["mn_round_restarted"] to claim the restart.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        ctx_extra = getattr(ctx, "extra", None) or {}
        # The session's wall-clock budget, resolved per round rather than once
        # per task: a baseline runs up to three of them, and a warmup that
        # overran has already spent budget the ones after it were counting on.
        _session_state = ctx_extra.get("shared_state") or self.shared_state
        session_deadline_sec, _ = session_grid_bounds(_session_state)
        timeout_sec = self._session_capped_timeout(timeout_sec, session_deadline_sec, output_dir=output_dir)
        if not ctx_extra.get("mn_round_restarted"):
            try:
                # Merge the reference base UNDER the per-task args (last-wins) so
                # a multi-node per-round restart carries the same reference flags
                # the single-node materialized YAML does.
                from ._grid_runner import merge_server_args
                from ._workload_envs import resolve_reference_base

                _mn_ref_args, _mn_ref_envs = resolve_reference_base()
                # Base on effective_extra_server_args (carries the one-shot
                # cuda-graph eager-fallback flag when armed) so the MN per-round
                # restart keeps that fallback too.
                _mn_task_args = effective_extra_server_args
                # Fold in the operator ``--server-args``
                # (``INFERENCE_OPTIMIZER_SERVER_ARGS``). Single-node and explore
                # variants apply it via the materialized YAML's ``EXTRA_*_ARGS``,
                # but the multi-node baseline server is launched by
                # launch_multinode (Magpie runs client-only) and never sees the
                # YAML, so without this the baseline runs a near-default server
                # while explore variants get the tuned flags — an unfair, skewed
                # baseline. Priority low->high: reference < operator < per-task.
                _mn_operator_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
                _mn_base = _mn_ref_args
                if _mn_operator_args:
                    _mn_base = merge_server_args(_mn_base, _mn_operator_args) if _mn_base else _mn_operator_args
                _mn_server_args = merge_server_args(_mn_base, _mn_task_args) if _mn_base else _mn_task_args
                _mn_env = {str(k): str(v) for k, v in _mn_ref_envs.items()}
                # PD knobs auto-resolved by the helper from $PD_* env, falling
                # back to state.json.
                await restart_server_for_round(
                    extra_server_args=_mn_server_args,
                    extra_env=_mn_env or None,
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
                    "output_dir": str(output_dir),
                }

        from ._multi_node_env import log_mn_banner

        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s", cmd, output_dir)

        # Magpie launched via ``run_with_session_kill`` so the whole descendant
        # tree is torn down on every exit path (plain subprocess.run leaks
        # daemonized server processes).
        # Multi-node client warmup: one discarded pass against the persistent
        # remote server (restarted just above) to warm JIT / steady-state
        # before the measured pass. Best-effort; MN-only; skipped when another
        # executor claimed the restart (profile round). Default ON
        # (INFERENCE_OPTIMIZER_MN_BENCH_WARMUP=0 disables).
        from ._multi_node_env import (
            is_multi_node as _mn_imn,
            mn_bench_warmup_enabled as _mn_warm,
        )

        round_warnings: list[str] = []
        if _mn_imn() and _mn_warm() and not ctx_extra.get("mn_round_restarted"):
            _mn_warm_result = await self._mn_warmup_pass(
                cmd=cmd,
                env=env,
                output_dir=output_dir,
                framework=framework,
                timeout_sec=timeout_sec,
                session_deadline_sec=session_deadline_sec,
                capture_meta=capture_meta,
                round_warnings=round_warnings,
                ctx_extra=ctx_extra,
            )
            if _mn_warm_result is not None:
                return _mn_warm_result

        workspaces_before = snapshot_workspaces(output_dir)
        subprocess_started_unix = time.time()
        # Anchor the Magpie parent process cwd to the per-task output_dir. NOTE:
        # this does NOT keep the server's cuda-graph dump safe on its own —
        # Magpie re-roots the actual server via ``cd <inferencex>``;
        # ``_ensure_local_inferencex`` above keeps that checkout on local disk.
        output_dir.mkdir(parents=True, exist_ok=True)
        # A reused output_dir may still hold a prior attempt's server.log, whose
        # terminal init markers would misclassify THIS attempt as
        # ``server_init_dead``. Clear it so classification only sees this
        # attempt's log.
        stale_server_log = output_dir / "server.log"
        try:
            stale_server_log.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning(
                "baseline_executor: could not clear stale server.log %s (%s); "
                "a prior attempt's markers may bias failure classification.",
                stale_server_log,
                exc,
            )
        # And the ready stamp beside it, for the same reason: a prior attempt's
        # would make this attempt's boot look like it never happened.
        clear_server_ready_stamp(str(stale_server_log))
        try:
            if serving_lease is not None:
                # Ray-managed GPU execution (§12 T1): run inside the lease's
                # actor (holds num_gpus across this run's rounds). Ray owns
                # *_VISIBLE_DEVICES, so strip the YAML device list first (T2).
                from ._ray_backend import strip_visible_devices_from_config

                ray_config_path = strip_visible_devices_from_config(config_path)
                ray_cmd = build_benchmark_command(
                    python_exe=self.magpie_python,
                    config_path=ray_config_path,
                    output_dir=output_dir,
                )
                # No liveness callback is possible here: the round runs inside a
                # Ray actor in another process (potentially on another node) and
                # only its final ``(rc, stdout, stderr)`` crosses back, so there
                # is nothing local to call per line of child output. A Ray-backed
                # round reports on entry and then goes quiet until it returns — a
                # known gap, not an oversight.
                proc_returncode, proc_stdout, proc_stderr = await asyncio.to_thread(
                    serving_lease.run_session_kill,
                    ray_cmd,
                    env=env,
                    cwd=str(output_dir),
                    timeout=timeout_sec,
                    server_log_path=watchdog_server_log,
                    session_remaining_sec=session_deadline_to_remaining_sec(session_deadline_sec),
                )
                subprocess_runtime_sec = max(0.0, time.time() - subprocess_started_unix)
            else:
                async with heartbeat_while_output_flows(
                    unit="baseline_round",
                    label="benchmark",
                ) as activity:
                    proc = await asyncio.to_thread(
                        run_with_session_kill,
                        cmd,
                        env=env,
                        cwd=str(output_dir),
                        timeout=timeout_sec,
                        server_log_path=watchdog_server_log,
                        on_output=activity.note,
                        session_deadline_sec=session_deadline_sec,
                    )
                subprocess_runtime_sec = max(
                    0.0,
                    time.time() - subprocess_started_unix,
                )
                proc_returncode = proc.returncode
                proc_stdout = proc.stdout
                proc_stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timeout_destination = select_run_workspace(output_dir, known_before=workspaces_before) or output_dir
            timeout_harvested = harvest_leaked_artifacts(
                timeout_destination,
                subprocess_started_unix=subprocess_started_unix,
            )
            return {
                "status": "failed",
                "error_class": "timeout",
                "error": f"baseline benchmark exceeded {timeout_sec}s: {exc}",
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in timeout_harvested],
                "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in timeout_harvested],
                **capture_meta,
            }

        stopped = stopped_by_the_run(proc_returncode)
        if stopped is not None:
            return _stopped_round_result(
                stopped,
                round_label="measured round",
                returncode=proc_returncode,
                runtime_sec=subprocess_runtime_sec,
                output_dir=output_dir,
                capture_meta=capture_meta,
            )

        # Detokenizer-stall watchdog reap: the server came up healthy but went
        # silent for the stall grace window (hung engine / wedged detokenizer).
        # A stall reap leaves no benchmark_* workspace; a distinct error_class
        # lets the coordinator fast-fail instead of burning the full timeout.
        if proc_returncode == DETOKENIZER_STALL_RETURNCODE:
            stall_destination = select_run_workspace(output_dir, known_before=workspaces_before) or output_dir
            stall_harvested = harvest_leaked_artifacts(
                stall_destination,
                subprocess_started_unix=subprocess_started_unix,
            )
            log.warning(
                "baseline_executor: detokenizer-stall watchdog reaped run "
                "(server ready but log went silent); error_class=detokenizer_stall."
            )
            return {
                "status": "failed",
                "error_class": "detokenizer_stall",
                "returncode": proc_returncode,
                "error": (
                    "server reported ready but emitted no log output (hung "
                    "engine / detokenizer stall); reaped by the "
                    "detokenizer-stall watchdog. See server.log."
                ),
                "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in stall_harvested],
                "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in stall_harvested],
                **capture_meta,
            }

        # When the server's engine/worker bootstrap dies, the root cause is in
        # server.log, not Magpie's stdout/stderr tail, and the liveness watchdog
        # may have reaped the hung parent with ``SERVER_DEAD_RETURNCODE``. Detect
        # that once here and reuse it across the failure branches so the failure
        # is classified ``server_init_dead``. Backend-agnostic (vLLM + SGLang).
        server_death_excerpt = server_log_death_excerpt(str(output_dir / "server.log"))
        server_init_dead = server_death_excerpt is not None or proc_returncode == SERVER_DEAD_RETURNCODE
        server_init_dead_error = server_death_excerpt or (
            "server engine/worker init failed (reaped by liveness watchdog); see server.log"
        )

        # Detect cuda-graph capture failures (OOM-rooted ones excluded).
        # Markers live in server.log; read a bounded tail for classification.
        server_log_tail = ""
        try:
            slog = output_dir / "server.log"
            if slog.exists():
                with open(slog, "rb") as f:
                    f.seek(0, 2)
                    sz = f.tell()
                    f.seek(max(0, sz - 65536))
                    server_log_tail = f.read().decode("utf-8", "replace")
        except OSError:
            server_log_tail = ""
        cuda_graph_capture_failed = _is_cuda_graph_capture_failure(
            server_log_tail,
            proc_stderr or "",
            proc_stdout or "",
        )

        workspace = select_run_workspace(output_dir, known_before=workspaces_before)
        # Always-on artifact harvest: copy wrapper-side leaks into the task
        # workspace so failure-path diagnostics survive; mtime gating rejects
        # stale prior-run leaks.
        harvest_destination = workspace if workspace is not None else output_dir
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=subprocess_started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if workspace is None:
            failure_extras = {
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in harvested],
                **capture_meta,
            }
            # Magpie never created a benchmark_* workspace, so the wrapper never
            # wrote server.log. Persist captured stderr/stdout so the failure
            # survives the NFS clone and S3 archive.
            captured = redact_secret_values((proc_stderr or "") + (proc_stdout or ""))
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
            # cuda-graph capture failures take priority over server_init_dead:
            # only this class arms the one-shot disable-cuda-graph retry.
            if cuda_graph_capture_failed:
                return {
                    "status": "failed",
                    "error_class": "cuda_graph_capture_failed",
                    "returncode": proc_returncode,
                    "error": redact_secret_values(
                        server_init_dead_error if server_init_dead else (proc_stderr or proc_stdout or "")[-2000:]
                    ),
                    **failure_extras,
                }
            if server_init_dead:
                return {
                    "status": "failed",
                    "error_class": "server_init_dead",
                    "returncode": proc_returncode,
                    "error": redact_secret_values(server_init_dead_error),
                    **failure_extras,
                }
            if proc_returncode != 0:
                tail = redact_secret_values((proc_stderr or proc_stdout or "")[-2000:])
                err_class = _classify_subprocess_error(
                    subprocess_runtime_sec,
                    tail,
                )
                return {
                    "status": "failed",
                    "error_class": err_class,
                    "returncode": proc_returncode,
                    "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                    "error": tail,
                    **failure_extras,
                }
            return {
                "status": "failed",
                "error_class": "no_workspace",
                "error": "Magpie completed but produced no benchmark_* workspace",
                **failure_extras,
            }
        report_path = workspace / "benchmark_report.json"
        report: dict[str, Any] | None = None
        if report_path.exists():
            try:
                with report_path.open(encoding="utf-8") as f:
                    loaded = json.load(f)
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                report = None

        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        warnings = round_warnings + list(measurement.pop("nonfatal_warnings", []) or [])
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")

        if not measurement.get("valid_measurement"):
            # cuda-graph capture failure wins over server_init_dead so the
            # one-shot disable-cuda-graph retry is armed even when both co-occur.
            if cuda_graph_capture_failed:
                error_class = "cuda_graph_capture_failed"
                error = server_init_dead_error if server_init_dead else ((proc_stderr or proc_stdout or "")[-2000:])
            elif server_init_dead:
                error_class = "server_init_dead"
                error = server_init_dead_error
            elif proc_returncode != 0:
                tail = (proc_stderr or proc_stdout or "")[-2000:]
                error_class = _classify_subprocess_error(
                    subprocess_runtime_sec,
                    tail,
                )
                error = tail
            elif not report_path.exists():
                error_class = "no_report"
                error = f"benchmark_report.json missing under {workspace}"
            else:
                error_class = "invalid_measurement"
                error = "benchmark report did not contain positive throughput and completed requests"
            error = redact_secret_values(error)
            return {
                "status": "failed",
                "error_class": error_class,
                "returncode": proc_returncode,
                "error": error,
                "output_dir": str(output_dir),
                "workspace": str(workspace),
                "report_path": str(report_path) if report_path.exists() else None,
                "reported_success": measurement.get("reported_success"),
                "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                "nonfatal_warnings": warnings,
                **capture_meta,
            }

        if proc_returncode != 0:
            return {
                "status": "failed",
                "error_class": "magpie_nonzero_after_valid_measurement",
                "returncode": proc_returncode,
                "error": redact_secret_values((proc_stderr or proc_stdout or "")[-2000:]),
                "output_dir": str(output_dir),
                "workspace": str(workspace),
                "report_path": str(report_path) if report_path.exists() else None,
                "reported_success": measurement.get("reported_success"),
                "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                "nonfatal_warnings": warnings,
                **capture_meta,
            }

        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": proc_returncode,
            "output_dir": str(output_dir),
            "result_dir": str(result_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Materialized YAML for THIS baseline. Coordinator promotes it into
            # SharedState.baseline_config_path so downstream tasks reuse it as
            # `config_path` (else variants render from smoke defaults).
            "materialized_config": str(materialized_config_path),
            # Magpie subprocess wall-clock (success path only). Coordinator
            # promotes into ``SharedState.baseline_runtime_sec``, the explore
            # overtime-kill anchor. Omitted on failure paths.
            "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
            # The benchmark's own share of that wall-clock, boot excluded. Kept
            # separate rather than folded in because the two are spent by
            # different things: every later variant boots again, so it is the
            # sum that prices one, while only this part prices a pass that
            # re-attaches. ``None`` when nothing recorded a ready boundary.
            "post_ready_runtime_sec": _round_post_ready_sec(
                watchdog_server_log,
                started_unix=subprocess_started_unix,
                runtime_sec=subprocess_runtime_sec,
            ),
            # Authoritative (materialized-config) view of whether the serving
            # lm-eval ran this run. The accuracy-stop decision reads this rather
            # than re-deriving from params, so a YAML/reference-env RUN_EVAL=false
            # is honored as an intentional opt-out.
            "run_eval_disabled": bool(run_eval_disabled),
        }

        # Parse accuracy eval results (GSM8K for serving, or the image-quality
        # gate for scriptable frameworks). Pass the framework so scriptable runs
        # fail closed on a missing quality gate instead of falling back to GSM8K.
        eval_framework = (report or {}).get("framework") or os.environ.get("FRAMEWORK") or None
        from hyperloom.inference_optimizer import framework_registry

        # RUN_EVAL gates ONLY the serving lm-eval GSM8K run. Scriptable (xDiT)
        # workloads carry no lm-eval; their sole correctness signal is the image
        # ``quality_gate`` embedded in ``benchmark_report.json``, which the bench
        # script writes every run and ``parse_quality_gate`` resolves by newest
        # mtime -- so there is no stale-artifact risk to guard against. The skip
        # below therefore applies to serving only, never dropping a scriptable
        # gate (which would leave baseline_accuracy=0 -> throughput-only KEEP).
        eval_scriptable = framework_registry.is_scriptable(eval_framework)
        if run_eval_disabled and not eval_scriptable:
            # Serving RUN_EVAL was off this run (eval-failure fallback or
            # ``disable_run_eval``), so lm-eval did not execute and there is no
            # fresh accuracy to read. Do NOT parse: the eval-failure retry reuses
            # ``output_dir``, so the slot may still hold a prior attempt's
            # ``results*.json`` and reading it would promote a stale score into
            # baseline_accuracy. Reading eval output strictly follows running eval.
            log.info(
                "baseline_executor: RUN_EVAL disabled this run (serving); skipping accuracy parse (no lm-eval executed)"
            )
        else:
            from ._accuracy_gate import eval_probe_summary, parse_eval_results, read_eval_probe

            # Search from ``$RESULT_DIR`` so serving runs survive benchmark_lib.sh
            # moving/cleaning ``$EVAL_RESULT_DIR`` and scriptable quality gates
            # still resolve from Magpie's benchmark reports.
            eval_search_root = result_dir
            eval_data = parse_eval_results(eval_search_root, framework=eval_framework)
            if eval_data.get("accuracy") is not None:
                result["accuracy"] = eval_data["accuracy"]
                result["accuracy_task"] = eval_data.get("task", "gsm8k")
                result["accuracy_metric"] = eval_data.get("metric", "")
                result["accuracy_source"] = eval_data.get("source_file", "")
                log.info("baseline_executor: accuracy=%.4f (%s)", result["accuracy"], result["accuracy_task"])
            else:
                log.warning("baseline_executor: accuracy eval not found: %s", eval_data.get("error", "unknown"))
            # Records why the score is ~0; the score itself is already correct.
            eval_probe = read_eval_probe(eval_search_root)
            if eval_probe:
                result["eval_probe"] = eval_probe
                log.warning("baseline_executor: %s", eval_probe_summary(eval_probe))

        log.info(
            "baseline_executor: %s %s (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            framework_registry.format_primary_metric(eval_framework, result["output_throughput"]),
            result["e2el_mean_ms"] or 0.0,
        )
        return result


baseline_executor = BaselineExecutor()


__all__ = [
    "AITER_JIT_PROBE_PATHS",
    "BASELINE_COLD_START_TIMEOUT_SEC",
    "AGENTX_BASELINE_OVERHEAD_SEC",
    "AGENTX_DEFAULT_DURATION_SEC",
    "BASELINE_DEFAULT_TIMEOUT_SEC",
    "agentx_baseline_timeout_sec",
    "BaselineExecutor",
    "COLD_START_KERNEL_THRESHOLD",
    "baseline_executor",
]
