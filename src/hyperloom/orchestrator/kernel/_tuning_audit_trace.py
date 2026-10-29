# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The audit row a GEMM-tuning run leaves behind, and why each tuner ended as it did."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from common.io import JsonLineFormat, append_jsonl

log = logging.getLogger(__name__)


# A tuner error is a diagnostic pointer, not the diagnosis: the full text lives
# in the run's own result.json and tune.log. 400 characters is enough to carry
# the argparse line or the aiter marker that says which of the two it was.
_TRACE_TUNER_ERROR_MAXLEN = 400

# Emitted even when null. ``kept`` is null on every row observed so far, and an
# absent key would be indistinguishable from ``false``.
_TRACE_TUNER_ALWAYS_KEYS = ("tuner", "best_micro_speedup", "kept")


def _trace_tuner_row(tuner: dict[str, Any]) -> dict[str, Any]:
    """One per-tuner entry for the audit row, keeping why it ended as it did.

    ``tuner``/``best_micro_speedup``/``kept`` alone cannot separate a tuner that
    crashed from one that ran and found nothing -- the single question the audit
    trail exists to answer. Without the reason, a tuner whose runs argparse
    rejected in seconds reads as a clean ``no_improvement`` (#1211).
    """
    error = tuner.get("error")
    if isinstance(error, str) and len(error) > _TRACE_TUNER_ERROR_MAXLEN:
        error = error[:_TRACE_TUNER_ERROR_MAXLEN] + "..."
    row = {
        "tuner": tuner.get("tuner") or tuner.get("name"),
        "best_micro_speedup": tuner.get("best_micro_speedup"),
        "kept": tuner.get("kept"),
        "status": tuner.get("status"),
        "elapsed_s": tuner.get("elapsed_s"),
        "error_class": tuner.get("error_class"),
        "error": error,
    }
    # A clean run stays as compact as before: everything added here is dropped
    # when it is null, so a successful row gains only status and elapsed_s.
    return {k: v for k, v in row.items() if k in _TRACE_TUNER_ALWAYS_KEYS or v is not None}


def _trace_gemm_tuning_run(result: Any, *, session_dir: Path) -> None:
    """Append one ``gemm_tuning.jsonl`` audit row for a GEMM-tuning run.

    Distils the run result into a compact source-attribution row (engine,
    decision, speedup, per-tuner summary) appended to
    ``reports/trace/gemm_tuning.jsonl``. Best-effort; any failure is swallowed.
    """
    if not isinstance(result, dict):
        return
    from datetime import datetime, timezone

    from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path

    engine = str(result.get("engine") or result.get("backend") or "").strip().lower() or "unknown"
    tuners: list[dict[str, Any]] = [
        _trace_tuner_row(t) for t in (result.get("tuners_run") or []) if isinstance(t, dict)
    ]
    # The envelope reported no error class even when a tuner had named one, so a
    # crashed run and a barren one looked alike at the top level too. Take the
    # first one a tuner supplied rather than leaving the field null.
    error_class = result.get("error_class") or next((t["error_class"] for t in tuners if t.get("error_class")), None)
    row = {
        "kind": "gemm_tuning",
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "engine": engine,
        "backend": result.get("backend"),
        "status": result.get("status"),
        "decision": result.get("decision"),
        "micro_decision": result.get("micro_decision"),
        "best_speedup": result.get("best_speedup"),
        "precision": result.get("precision"),
        "framework": result.get("framework"),
        "gpu_type": result.get("gpu_type"),
        "tuned_file": result.get("tuned_file"),
        "workspace": result.get("workspace"),
        "requires_e2e_validation": result.get("requires_e2e_validation"),
        "tuners_run": tuners,
        # A crash stays in the audit even when a sibling tuner won and the run was
        # kept -- otherwise a KEEP row hid the failures behind it.
        "failed_tuners": result.get("failed_tuners") or None,
        "error_class": error_class,
    }
    row = {k: v for k, v in row.items() if v is not None}
    try:
        append_jsonl(gemm_tuning_steps_path(session_dir), row, make_parents=True, fmt=JsonLineFormat(sort_keys=True))
    except OSError:
        log.debug("full-trace: gemm_tuning audit append failed", exc_info=True)
