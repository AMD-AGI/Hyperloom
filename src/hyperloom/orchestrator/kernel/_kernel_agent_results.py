# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read what kernel-agent recorded for one kernel: its result file and backend ladder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


#: ``backend_ladder[].error_class`` vocabulary surfaced into
#: ``failure_reason_breakdown``. Empty string is reserved for succeeded attempts.
ERROR_CLASS_TIMEOUT = "timeout"
ERROR_CLASS_PREPROCESS_FAILED = "preprocess_failed"
ERROR_CLASS_COMPILE_FAILED = "compile_failed"
ERROR_CLASS_CORRECTNESS_FAILED = "correctness_failed"
ERROR_CLASS_AGENT_ERROR = "agent_error"
ERROR_CLASS_UNKNOWN = "unknown"

#: On early failure kernel-agent points ``optimized_path`` at a stdout/stderr
#: dump; those must not flip ``produced_artifact=true``.
_ARTIFACT_LOG_SUFFIXES = (
    "_stdout.log",
    "_stderr.log",
    ".log",
    ".txt",
)


def _is_real_artifact_path(path: str) -> bool:
    """True only when ``path`` looks like a real kernel artifact."""
    if not path:
        return False
    p = path.strip()
    if not p:
        return False
    low = p.lower()
    if any(low.endswith(suf) for suf in _ARTIFACT_LOG_SUFFIXES):
        return False
    fname = low.rsplit("/", 1)[-1]
    if "_stdout" in fname or "_stderr" in fname:
        return False
    return True


_RE_TIMEOUT = re.compile(r"Timed out after (\d+)s")
# stdout_tail is column-wrapped, so the signal can straddle newlines.
_RE_PREPROCESS_FAILED = re.compile(
    r"preprocess[\s\S]{0,300}?success=False"
    r"(?:[\s\S]{0,80}?errors=(\d+))?",
    re.IGNORECASE,
)
_RE_COMPILE_FAILED = re.compile(
    r"(compile|build).{0,30}(failed|error)|undefined reference",
    re.IGNORECASE,
)
_RE_CORRECTNESS_FAILED = re.compile(
    r"correctness.{0,30}(failed|mismatch)|accuracy mismatch",
    re.IGNORECASE,
)


def _classify_attempt_failure(
    attempt: dict[str, Any],
) -> tuple[str, str]:
    """Classify a failed/partial attempt into ``(error_class, error_message)``."""
    status = str(attempt.get("status") or "").strip().lower()
    if status == "succeeded":
        return "", ""
    stdout = str(attempt.get("stdout_tail") or "")
    explicit_err = str(attempt.get("error_message") or "")

    for blob in (explicit_err, stdout):
        m = _RE_TIMEOUT.search(blob)
        if m:
            secs = m.group(1)
            return ERROR_CLASS_TIMEOUT, f"Timed out after {secs}s"

    m = _RE_PREPROCESS_FAILED.search(stdout)
    if m:
        errs = m.group(1) or "?"
        return (
            ERROR_CLASS_PREPROCESS_FAILED,
            f"preprocess reported {errs} error(s)",
        )

    if _RE_COMPILE_FAILED.search(stdout):
        return ERROR_CLASS_COMPILE_FAILED, "compilation failed"

    if _RE_CORRECTNESS_FAILED.search(stdout):
        return ERROR_CLASS_CORRECTNESS_FAILED, "correctness check failed"

    rc = attempt.get("returncode")
    if isinstance(rc, int) and rc != 0:
        return ERROR_CLASS_AGENT_ERROR, f"agent exit code {rc}"

    return ERROR_CLASS_UNKNOWN, ""


def _backend_results_dir(session_dir: Path, session_id: str) -> Path | None:
    """Return ``<sd>/kernel-agent/runs/<key>/results`` or ``None``."""
    from hyperloom.inference_optimizer.session.session_paths import kernel_agent_runs_root

    runs_root = kernel_agent_runs_root(Path(session_dir))
    if not runs_root.is_dir():
        return None
    for key in (session_dir.name, str(session_id or "").strip()):
        if not key:
            continue
        candidate = runs_root / key / "results"
        if candidate.is_dir():
            return candidate
    subdirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0] / "results"
        if candidate.is_dir():
            return candidate
    return None


def _load_kernel_result(
    results_dir: Path | None,
    kernel_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Read the raw kernel-agent ``results/<kid>.json`` payload."""
    if results_dir is None:
        return None, "kernel_agent_results_dir_missing"
    fpath = results_dir / f"{kernel_id}.json"
    if not fpath.is_file():
        return None, "kernel_agent_result_file_missing"
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "parse_error"
    if not isinstance(data, dict):
        return None, "parse_error"
    return data, ""


def _load_backend_ladder(
    results_dir: Path | None,
    kernel_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Parse one kernel's kernel-agent ``results/<kid>.json`` attempts."""
    data, reason = _load_kernel_result(results_dir, kernel_id)
    if data is None:
        return [], reason
    raw_attempts = data.get("attempts") or []
    if not isinstance(raw_attempts, list) or not raw_attempts:
        return [], "no_attempts_recorded"
    ladder: list[dict[str, Any]] = []
    for a in raw_attempts:
        if not isinstance(a, dict):
            continue
        produced = _is_real_artifact_path(a.get("optimized_path") or "")
        row: dict[str, Any] = {
            "backend": str(a.get("backend") or ""),
            "status": str(a.get("status") or ""),
            "attempt_id": str(a.get("attempt_id") or ""),
            "produced_artifact": produced,
        }
        # Backend self-skip marker for the outcome classifier.
        if a.get("skipped"):
            row["skipped"] = True
        elapsed = a.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            row["elapsed_sec"] = float(elapsed)
        err_class, err_msg = _classify_attempt_failure(a)
        if err_class:
            row["error_class"] = err_class
        if err_msg:
            row["error_message"] = err_msg
        ladder.append(row)
    return ladder, ""
