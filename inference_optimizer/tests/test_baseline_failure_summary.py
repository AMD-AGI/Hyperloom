# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``baseline_failed`` root-cause surfacing tests (issue #465).

On ``baseline_failed`` the top-level ``final.json`` / report must headline the
real terminal engine/worker fault from the last failed baseline attempt, not a
benign upstream WARN (e.g. transformers' ``modeling_cohere2.py`` ENOENT).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from inference_optimizer.orchestrator.action_executors.report import (
    _build_summary_dict,
    _classify_root_cause_type,
    _format_md,
    _highlight_is_benign,
    _is_benign_failure_text,
)

_BENIGN_WARN = (
    "[WARN] Error: [Errno 2] No such file or directory: "
    "'/app/.../transformers/models/cohere2/modeling_cohere2.py'"
)

_SERVER_LOG_OOM = (
    "non-default args: {'tensor_parallel_size': 8}\n"
    "[aiter] import [module_aiter_core] ...\n"
    "[2] [FATAL ERROR]: HIP failure: 'out of memory' "
    "(in ncclCommInitRank during init_model_parallel)\n"
    "RuntimeError: NCCL error: unhandled cuda error\n"
    "EngineCore failed to start.\n"
)


def _mock_state(*, stop_reason: str = "baseline_failed") -> SimpleNamespace:
    """Minimal SharedState-shaped object for ``_build_summary_dict``."""
    return SimpleNamespace(
        session_id="sid",
        model_name="command-a-plus-05-2026-fp8",
        model_path="/tmp/model",
        model_class="moe",
        stop_reason=stop_reason,
        baseline_tput=0.0,
        baseline_accuracy=0.0,
        last_remaining_gaps_assessment={},
        remaining_gaps_assessments=[],
        current_best={},
        cumulative_gain=0.0,
        cumulative_gain_validated=0.0,
        cumulative_gain_validated_ts="",
        cumulative_gain_validated_stack_len=0,
        optimization_stack=[],
        crash_count=2,
        pruned_families=[],
        max_minutes=720,
        roofline_snapshots=[],
    )


def _write_baseline_attempt(
    session_dir: Path,
    task_id: str,
    *,
    error_class: str,
    error: str,
    server_log: str | None = None,
) -> Path:
    """Materialise ``runs/baseline/<task_id>/result.json`` (+ optional log)."""
    task_dir = session_dir / "runs" / "baseline" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "failed",
        "error_class": error_class,
        "error": error,
        "output_dir": str(task_dir),
    }
    (task_dir / "result.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    if server_log is not None:
        (task_dir / "server.log").write_text(server_log, encoding="utf-8")
    return task_dir


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_is_benign_failure_text():
    assert _is_benign_failure_text(_BENIGN_WARN)
    assert not _is_benign_failure_text("HIP out of memory during ncclCommInitRank")


def test_highlight_is_benign_checks_summary_and_payload():
    assert _highlight_is_benign({"summary": _BENIGN_WARN, "payload": {}})
    assert _highlight_is_benign(
        {"summary": "sev=high", "payload": {"detail": _BENIGN_WARN}}
    )
    assert not _highlight_is_benign(
        {"summary": "sev=high EngineCore failed", "payload": {"x": 1}}
    )


def test_classify_root_cause_type():
    assert _classify_root_cause_type("server_init_dead", "HIP out of memory") == "oom"
    assert _classify_root_cause_type("timeout", "baseline benchmark exceeded 600s") == "benchmark_timeout"
    assert _classify_root_cause_type("server_init_dead", "EngineCore failed to start") == "engine_core_init"
    assert _classify_root_cause_type("subprocess_nonzero", "boom") == "worker_crash"
    assert _classify_root_cause_type("unknown", "") == "unknown"


# ---------------------------------------------------------------------------
# failure_summary from result.json
# ---------------------------------------------------------------------------
def test_failure_summary_from_result_error(tmp_path):
    """A server_init_dead attempt's excerpt becomes the headline root cause."""
    _write_baseline_attempt(
        tmp_path, "attempt1",
        error_class="server_init_dead",
        error=_SERVER_LOG_OOM,
    )
    state = _mock_state()
    summary = _build_summary_dict(
        state, ev_counts={}, highlights=[], session_dir=tmp_path,
    )
    fs = summary["failure_summary"]
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert fs["last_attempt_id"] == "attempt1"


def test_failure_summary_suppresses_benign_warn_and_falls_back_to_log(tmp_path):
    """Benign WARN in result.error → fall back to server.log terminal marker."""
    _write_baseline_attempt(
        tmp_path, "attempt2",
        error_class="unknown",
        error=_BENIGN_WARN,
        server_log=_SERVER_LOG_OOM,
    )
    state = _mock_state()
    summary = _build_summary_dict(
        state, ev_counts={}, highlights=[], session_dir=tmp_path,
    )
    fs = summary["failure_summary"]
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert "modeling_cohere2.py" not in fs["root_cause"]
    assert fs["suppressed_benign"]
    assert fs["server_log"].endswith("server.log")


def test_failure_summary_picks_latest_failed_attempt(tmp_path):
    """Most recent failed attempt wins when several are present."""
    import os
    import time

    first = _write_baseline_attempt(
        tmp_path, "attempt_old",
        error_class="timeout",
        error="baseline benchmark exceeded 600s",
    )
    time.sleep(0.01)
    _write_baseline_attempt(
        tmp_path, "attempt_new",
        error_class="server_init_dead",
        error=_SERVER_LOG_OOM,
    )
    # Make the ordering deterministic regardless of FS mtime granularity.
    old = time.time() - 100
    os.utime(first / "result.json", (old, old))

    state = _mock_state()
    summary = _build_summary_dict(
        state, ev_counts={}, highlights=[], session_dir=tmp_path,
    )
    assert summary["failure_summary"]["last_attempt_id"] == "attempt_new"


def test_failure_summary_absent_when_not_baseline_failed(tmp_path):
    """Non-failure stop reasons never attach a failure_summary."""
    _write_baseline_attempt(
        tmp_path, "attempt1",
        error_class="server_init_dead",
        error=_SERVER_LOG_OOM,
    )
    state = _mock_state(stop_reason="time_exhausted")
    summary = _build_summary_dict(
        state, ev_counts={}, highlights=[], session_dir=tmp_path,
    )
    assert "failure_summary" not in summary


def test_failure_summary_absent_without_session_dir(tmp_path):
    """Back-compat: no session_dir → no failure_summary (legacy callers)."""
    state = _mock_state()
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert "failure_summary" not in summary


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def test_format_md_surfaces_root_cause(tmp_path):
    _write_baseline_attempt(
        tmp_path, "attempt1",
        error_class="server_init_dead",
        error=_SERVER_LOG_OOM,
    )
    state = _mock_state()
    summary = _build_summary_dict(
        state, ev_counts={}, highlights=[], session_dir=tmp_path,
    )
    md = _format_md(summary)
    assert "Root cause" in md
    assert "oom" in md
    assert "modeling_cohere2.py" not in md
