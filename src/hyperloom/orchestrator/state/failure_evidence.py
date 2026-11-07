# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure helpers for structured variant failure evidence.

This module owns the vocabulary a failed attempt is named with: the error
classes, the text that is evidence of a cause, and the helpers that read one
out of the other. The executors that produce a failure and the report that
reads it back share these names from here, so a class is spelled once.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from inference_suite.contracts import FailedStep

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.failure_signature import SERVE_FLAG
from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID

FAILURE_STAGE_WARMUP: str = "warmup"
FAILURE_STAGE_DECISION: str = "decision"

FAILURE_ATTRIBUTION_CANDIDATE: str = "candidate_caused"
FAILURE_ATTRIBUTION_ENVIRONMENT: str = "environment"
FAILURE_ATTRIBUTION_HARNESS: str = "harness"
FAILURE_ATTRIBUTION_UNKNOWN: str = "unknown"
FAILURE_ATTRIBUTIONS: frozenset[str] = frozenset(
    {
        FAILURE_ATTRIBUTION_CANDIDATE,
        FAILURE_ATTRIBUTION_ENVIRONMENT,
        FAILURE_ATTRIBUTION_HARNESS,
        FAILURE_ATTRIBUTION_UNKNOWN,
    }
)

# Variant outcomes that produced no usable measurement, so they need evidence.
# ``REVERT`` is deliberately absent: it was measured, just not promoted.
UNMEASURED_OUTCOMES: frozenset[str] = frozenset({"FAILED", "KILLED_OVERTIME"})

# The error class of a subprocess that exited nonzero for no recognised cause.
SUBPROCESS_NONZERO: str = "subprocess_nonzero"
# The error class of a Magpie benchmark that served its whole protocol and still left no
# usable measurement behind: the round ran to the end, so the nonzero exit came from the
# harness around it. Archived ledgers carry it, and so do the Magpie rounds until that path goes.
MAGPIE_NONZERO_INVALID_MEASUREMENT: str = "magpie_nonzero_invalid_measurement"
SERVER_INIT_DEAD: str = "server_init_dead"
DETOKENIZER_STALL: str = "detokenizer_stall"
BENCHMARK_REPORT_MISSING: str = "benchmark_report_missing"
MAGPIE_TIMEOUT: str = "magpie_timeout"
YAML_BUILD_ERROR: str = "yaml_build_error"

# The class each inference_suite failed step is recorded under. A step that replaces a
# failure Hyperloom names independently of the runner keeps that class, so readers of the
# ledgers see one class per cause; a new step, or one replacing a Magpie class, is its own.
# The steps whose class an action module owns (the run's own stops, the speculative
# pin) are mapped beside that owner in the grid runner.
FAILED_STEP_ERROR_CLASSES: Mapping[FailedStep, str] = {
    FailedStep.CONFIG_INVALID: YAML_BUILD_ERROR,
    FailedStep.BASE_DIGEST_MISMATCH: FailedStep.BASE_DIGEST_MISMATCH.value,
    FailedStep.FRAMEWORK_VERSION_TOO_OLD: FailedStep.FRAMEWORK_VERSION_TOO_OLD.value,
    FailedStep.FRAMEWORK_VERSION_UNKNOWN: FailedStep.FRAMEWORK_VERSION_UNKNOWN.value,
    FailedStep.REUSE_NOT_OWNED: FailedStep.REUSE_NOT_OWNED.value,
    FailedStep.EVAL_CONTEXT_TOO_SHORT: FailedStep.EVAL_CONTEXT_TOO_SHORT.value,
    FailedStep.STAGING_MISSING: FailedStep.STAGING_MISSING.value,
    FailedStep.SOURCE_FILES_CHANGED: FailedStep.SOURCE_FILES_CHANGED.value,
    FailedStep.FLAG_UNKNOWN: ARGV_INVALID,
    FailedStep.GOLDEN_OWNED_TOUCHED: SERVE_FLAG,
    FailedStep.PORT_IN_USE: FailedStep.PORT_IN_USE.value,
    FailedStep.SERVER_EXITED_BEFORE_READY: SERVER_INIT_DEAD,
    FailedStep.SERVER_SILENT_BEFORE_READY: FailedStep.SERVER_SILENT_BEFORE_READY.value,
    FailedStep.SERVER_BOOT_TIMEOUT: FailedStep.SERVER_BOOT_TIMEOUT.value,
    FailedStep.SERVER_STALLED: DETOKENIZER_STALL,
    FailedStep.SERVER_DIED: FailedStep.SERVER_DIED.value,
    FailedStep.CLIENT_FAILED: FailedStep.CLIENT_FAILED.value,
    FailedStep.CLIENT_OUTPUT_MISSING: BENCHMARK_REPORT_MISSING,
    FailedStep.EVAL_FAILED: FailedStep.EVAL_FAILED.value,
    FailedStep.ROUND_TIMEOUT: FailedStep.ROUND_TIMEOUT.value,
    FailedStep.DISK_FULL: FailedStep.DISK_FULL.value,
    FailedStep.TEARDOWN_INCOMPLETE: FailedStep.TEARDOWN_INCOMPLETE.value,
}

# Matches every character replaced by ``_`` in a variant-name slug.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")
_CANDIDATE_ERROR_CLASSES = frozenset(
    {
        "capability_unsupported",
        "invalid_argument",
        "unsupported_config",
        SERVE_FLAG,
    }
)
_ENVIRONMENT_ERROR_CLASSES = frozenset(
    {
        "mn_server_restart_failed",
        FailedStep.DISK_FULL.value,
        FailedStep.FRAMEWORK_VERSION_TOO_OLD.value,
        FailedStep.FRAMEWORK_VERSION_UNKNOWN.value,
        FailedStep.PORT_IN_USE.value,
        FailedStep.REUSE_NOT_OWNED.value,
        FailedStep.SERVER_SILENT_BEFORE_READY.value,
        FailedStep.TEARDOWN_INCOMPLETE.value,
    }
)
_HARNESS_ERROR_CLASSES = frozenset(
    {
        "benchmark_cap_error",
        "benchmark_report_invalid_metric",
        BENCHMARK_REPORT_MISSING,
        "eval_probe_unpatchable",
        "inferencex_checkout_invalid",
        "magpie_nonzero_after_valid_measurement",
        MAGPIE_NONZERO_INVALID_MEASUREMENT,
        MAGPIE_TIMEOUT,
        "no_benchmark_workspace",
        "warmup_magpie_timeout",
        "warmup_round_failed",
        "warmup_yaml_build_error",
        YAML_BUILD_ERROR,
        FailedStep.BASE_DIGEST_MISMATCH.value,
        FailedStep.CLIENT_FAILED.value,
        FailedStep.EVAL_CONTEXT_TOO_SHORT.value,
        FailedStep.ROUND_TIMEOUT.value,
        FailedStep.SOURCE_FILES_CHANGED.value,
        FailedStep.STAGING_MISSING.value,
    }
)
_ENVIRONMENT_EVIDENCE_RE = re.compile(
    r"(?:concurrent|foreign|orphan(?:ed)?|leak(?:ed)?)\s+(?:gpu\s+)?(?:holder|process|engine)"
    r"|card\s+(?:is\s+)?(?:not\s+)?(?:clean|idle)"
    r"|(?:gpu|device)\s+(?:is\s+)?busy"
    r"|address\s+already\s+in\s+use"
    r"|port\s+\d+\s+is\s+unavailable"
    r"|temporary\s+failure\s+in\s+name\s+resolution",
    re.IGNORECASE,
)
# A server or its actor that died before serving: Magpie's health-wait errors and the serving lease's actor death.
_SERVER_DEATH_EVIDENCE_RE = re.compile(
    r"exited\s+(?:before\s+becoming\s+healthy|during\s+its\s+health\s+check)|ray_actor_error",
    re.IGNORECASE,
)
_CANDIDATE_EVIDENCE_RE = re.compile(
    r"unrecogni[sz]ed\s+(?:argument|option)"
    r"|unknown\s+(?:argument|option)"
    r"|invalid\s+choice"
    r"|unsupported\s+(?:argument|backend|configuration|dtype|option)",
    re.IGNORECASE,
)

# KV-cache OOM: weights loaded but no room left for the KV cache.
KV_CACHE_OOM_MARKERS: tuple[str, ...] = (
    "no gpu memory for the kv cache",
    "leave no gpu memory",
    "raise --mem-fraction-static above",
)

# Text in which a process says it died. Each marker names the death itself, not
# the subsystem around it: ranks, collectives and worker ids are printed by every
# healthy multi-GPU launch, so matching those reports a crash for a run that only
# exited nonzero. A WARN is chatter even when NCCL prints it; only an error, a
# timeout, a fault or an abort is a death.
WORKER_DEATH_MARKERS: tuple[str, ...] = (
    "nccl error",
    "nccl timeout",
    "unhandled system error",
    "watchdog caught",
    "worker crashed",
    "worker died",
    "worker exited",
    "worker process died",
    "killed by signal",
    "terminated by signal",
    "segmentation fault",
    "sigsegv",
    "sigabrt",
    "core dumped",
    # How a death ends up worded on this hardware: a GPU page fault from the ROCm
    # runtime, an aborted interpreter, an invalid access reported by the device
    # runtime, and an unhandled C++ exception, each of which ends the process.
    "memory access fault",
    "illegal memory access",
    "fatal python error",
    "terminate called after throwing",
)

# A negative exit status is how a launcher reports that a signal ended the process
# (torchrun prints ``exitcode: -6`` for SIGABRT), whichever signal it was.
_SIGNAL_EXIT_RE = re.compile(r"exit\s*code\s*[:=]\s*-\d+")

_SIGNATURE_VOLATILE_RE = re.compile(r"0x[0-9a-f]+|[0-9a-f]{8,}|\d+")
# The classes whose failures are told apart by their output. Each is given to a process
# whose exit Hyperloom cannot explain, so the output is all there is to compare; the other
# classes carry a fixed sentence that two unrelated failures share.
_SIGNED_FAILURE_CLASSES = frozenset(
    {SUBPROCESS_NONZERO, MAGPIE_NONZERO_INVALID_MEASUREMENT, FailedStep.CLIENT_FAILED.value}
)


def tail_excerpt(value: Any, *, limit: int = 1200) -> str | None:
    """Return the trailing ``limit`` characters of ``value`` after redaction."""
    if value is None:
        return None
    text = redact_secret_values(str(value))
    if not text:
        return None
    return text[-limit:] if len(text) > limit else text


def make_failure_id(*, task_id: str, fingerprint: str, variant_name: str = "") -> str:
    """Compute a failure id; must stay recomputable from the same inputs."""
    fp = (fingerprint or "").strip()
    key = fp[:12] if fp else _SLUG_RE.sub("_", (variant_name or "unknown"))[:12]
    return f"fail.{task_id}.{key}"


def classify_failure_attribution(
    *,
    error_class: Any,
    error_excerpt: Any = "",
    reason: Any = "",
    explicit: Any = "",
) -> str:
    """Classify whether an unmeasured failure says anything about the candidate.

    Classification is deliberately conservative. Generic launch, timeout, and
    benchmark failures remain outside the Experience corpus unless the producer
    supplies candidate-specific evidence. The full failure still remains in
    SBD V6 regardless of this result.
    """
    authored = str(explicit or "").strip().lower()
    if authored in FAILURE_ATTRIBUTIONS:
        return authored

    kind = str(error_class or "").strip().lower()
    evidence = f"{reason or ''}\n{error_excerpt or ''}"
    if kind in _ENVIRONMENT_ERROR_CLASSES or _ENVIRONMENT_EVIDENCE_RE.search(evidence):
        return FAILURE_ATTRIBUTION_ENVIRONMENT
    if kind in _CANDIDATE_ERROR_CLASSES or _CANDIDATE_EVIDENCE_RE.search(evidence):
        return FAILURE_ATTRIBUTION_CANDIDATE
    if kind in _HARNESS_ERROR_CLASSES and not _server_died(evidence):
        return FAILURE_ATTRIBUTION_HARNESS
    return FAILURE_ATTRIBUTION_UNKNOWN


def _server_died(evidence: str) -> bool:
    """Whether ``evidence`` shows the server process died, which no harness class explains."""
    lowered = evidence.lower()
    return bool(
        _SERVER_DEATH_EVIDENCE_RE.search(evidence)
        or any(marker in lowered for marker in WORKER_DEATH_MARKERS)
        or _SIGNAL_EXIT_RE.search(lowered)
    )


def classify_root_cause_type(error_class: str, error_text: str) -> str:
    """Map a baseline attempt's ``error_class`` + message to a coarse enum.

    Every branch names what the attempt left evidence of. ``subprocess_nonzero``
    is the producer's name for an exit it did not recognise, and a line that
    merely mentions a worker or NCCL is startup chatter, so neither is a crash:
    without evidence of a process dying the answer is ``unknown``, which leaves
    the excerpt as the headline rather than replacing it with a diagnosis the
    run then acts on.
    """
    blob = f"{error_class} {error_text}".lower()
    if error_class == "kv_cache_oom" or any(m in blob for m in KV_CACHE_OOM_MARKERS):
        return "kv_cache_oom"
    if "out of memory" in blob or "hip oom" in blob:
        return "oom"
    if error_class == "timeout" or "benchmark exceeded" in blob:
        return "benchmark_timeout"
    if (
        error_class == SERVER_INIT_DEAD
        or "engine core" in blob
        or "enginecore" in blob
        or "workerproc" in blob
        or "engine process failed" in blob
    ):
        return "engine_core_init"
    if any(marker in blob for marker in WORKER_DEATH_MARKERS) or _SIGNAL_EXIT_RE.search(blob):
        return "worker_crash"
    if classify_failure_attribution(error_class=error_class) == FAILURE_ATTRIBUTION_HARNESS:
        return "benchmark_harness"
    return "unknown"


def nonzero_output_signature(err_class: str, proc_stdout: str | None, proc_stderr: str | None) -> str:
    """Identify an unexplained nonzero exit by the process's whole output, ignoring numbers.

    The result's ``error`` is only the last 2000 characters of one stream, and two
    different causes (a HIP OOM, an NCCL timeout) can end in the same engine-death
    traceback that long, or differ only in the stream ``error`` leaves out, so the
    signature is taken from both full, redacted streams instead. Numbers (pids,
    ports, timestamps, durations) and hex ids differ between two attempts that failed
    for the same reason, so they are folded away.

    Args:
        err_class: The class the failure was given; only a class in
            :data:`_SIGNED_FAILURE_CLASSES` gets a signature, because other classes carry
            a fixed sentence that unrelated failures share.
        proc_stdout: The process's whole stdout.
        proc_stderr: The process's whole stderr.

    Returns:
        ``"<class>|<sha256>"``, or ``""`` when there is no signature (another class, or
        no output: two failures that said nothing are not known to be the same failure).
    """
    if err_class not in _SIGNED_FAILURE_CLASSES:
        return ""
    text = redact_secret_values(f"{proc_stderr or ''}\n{proc_stdout or ''}").strip().lower()
    if not text:
        return ""
    text = " ".join(_SIGNATURE_VOLATILE_RE.sub("#", text).split())
    return f"{err_class}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def failure_from_variant_outcome(
    *,
    task_id: str,
    round_id: str,
    vo: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure evidence packet from a per-variant-outcome row, carrying the attribution its producer classified."""
    fp = str(vo.get("fingerprint") or "")
    variant_name = str(vo.get("variant_name") or "")
    variant = vo.get("variant") or {}
    return {
        "failure_id": make_failure_id(task_id=task_id, fingerprint=fp, variant_name=variant_name),
        "task_id": task_id,
        "round_id": round_id,
        "variant_name": variant_name,
        "fingerprint": fp,
        "stage": str(vo.get("stage") or FAILURE_STAGE_DECISION),
        "outcome": str(vo.get("outcome") or ""),
        "error_class": str(vo.get("error_class") or ""),
        "attribution": str(vo.get("failure_attribution") or ""),
        "error_excerpt": vo.get("error_excerpt") or "",
        "reason": str(vo.get("reason") or ""),
        "server_log_path": vo.get("server_log_path"),
        "workspace": vo.get("workspace"),
        "raw_result_path": vo.get("raw_result_path"),
        "variant": {
            "extra_server_args": str(variant.get("extra_server_args") or ""),
            "extra_envs": dict(variant.get("extra_envs") or {}),
            "note": str(variant.get("note") or ""),
            "reasoning_origin": str(variant.get("reasoning_origin") or ""),
        },
    }


def render_failure_line(fe: dict[str, Any], *, excerpt_chars: int = 160) -> str:
    """Format one failure evidence packet as a compact single line."""
    error_class = str(fe.get("error_class") or "")
    body = str(fe.get("error_excerpt") or fe.get("reason") or "")
    parts = [
        f"fid={fe.get('failure_id') or ''}",
        f"variant={str(fe.get('variant_name') or '')!r}",
        f"stage={fe.get('stage') or ''}",
    ]
    if error_class:
        parts.append(f"err={error_class}")
    if body:
        parts.append(f"msg={body[-excerpt_chars:]!r}")
    return " ".join(parts)
