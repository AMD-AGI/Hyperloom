# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bring-up attempt logs and the boot-failure classification of a benchmark round."""

from __future__ import annotations

import gzip
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hyperloom.inference_optimizer.grid_server_args import with_capture_disabled
from ....state.failure_evidence import KV_CACHE_OOM_MARKERS, SUBPROCESS_NONZERO, engine_capacity_cause
from .._aiter_jit import is_aiter_jit_registry_mismatch
from .._subprocess_kill import server_log_death_excerpt
from .records import _BenchProcess

log = logging.getLogger(__name__)

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
    from ....bringup import DEGRADED_UNREADABLE

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


def _boot_failure(finished: _BenchProcess, server_log: Path) -> tuple[str, str] | None:
    death = server_log_death_excerpt(str(server_log))
    error = death if death is not None else (finished.stderr or finished.stdout)[-2000:]
    texts = (finished.server_log_text, finished.stderr, finished.stdout)
    if is_aiter_jit_registry_mismatch(*texts):
        return "aiter_jit_registry_mismatch", error
    if _is_cuda_graph_capture_failure(*texts):
        return "cuda_graph_capture_failed", error
    if death is None:
        return None
    # The card holds less than the launch settings ask for: a config limit, not an engine fault.
    return ("engine_capacity" if engine_capacity_cause(death) else "server_init_dead", death)
