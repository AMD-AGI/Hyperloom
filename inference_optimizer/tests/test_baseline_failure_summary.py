# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``baseline_failed`` root-cause surfacing tests (issue #465).

On ``baseline_failed`` the top-level ``final.json`` / report must headline the
real terminal engine/worker fault from the last failed baseline attempt, not a
benign upstream WARN (e.g. transformers' ``modeling_cohere2.py`` ENOENT).

These exercise the real persistence path: a failed baseline result is recorded
into ``SharedState`` via :meth:`SharedState.record_action_attempt` /
:meth:`SharedState.record_action_failure` (exactly as the Coordinator does in
``_handle_unpromotable_result``), and the report layer reads it back from state
— there is no on-disk ``runs/baseline/<task_id>/result.json``.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.action_executors.report import (
    _build_failure_summary,
    _build_summary_dict,
    _classify_root_cause_type,
    _format_md,
    _highlight_is_benign,
    _is_benign_failure_text,
    _partition_benign_lines,
)
from inference_optimizer.orchestrator.shared_state import SharedState

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

# A subprocess_nonzero stderr tail that mixes the benign WARN with the real
# fault on separate lines (the #465 reproduction).
_MIXED_STDERR = (
    f"{_BENIGN_WARN}\n"
    "Loading safetensors checkpoint shards: 100% complete\n"
    "[2] [FATAL ERROR]: HIP failure: 'out of memory' during ncclCommInitRank\n"
)


def _failed_state(
    *,
    error_class: str,
    error: str,
    workspace: str | None = None,
    task_id: str = "baseline-1",
    stop_reason: str = "baseline_failed",
) -> SharedState:
    """Build a SharedState carrying one failed baseline attempt, like the
    Coordinator's ``_handle_unpromotable_result`` does."""
    state = SharedState()
    state.set_stop_reason(stop_reason)
    result = {
        "status": "failed",
        "error_class": error_class,
        "error": error,
    }
    if workspace is not None:
        result["workspace"] = workspace
    state.record_action_attempt(
        action="baseline",
        task_id=task_id,
        status="failed",
        decision="no_promote",
        result=result,
    )
    state.record_action_failure(
        action="baseline", task_id=task_id, result=result,
    )
    return state


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_is_benign_failure_text():
    assert _is_benign_failure_text(_BENIGN_WARN)
    assert not _is_benign_failure_text("HIP out of memory during ncclCommInitRank")


def test_highlight_is_benign_uses_summary_only():
    # Benign headline → suppressed.
    assert _highlight_is_benign({"summary": _BENIGN_WARN, "payload": {}})
    # Real fault headline is kept even when payload mentions the benign file.
    assert not _highlight_is_benign(
        {"summary": "sev=high EngineCore failed", "payload": {"f": _BENIGN_WARN}}
    )


def test_partition_benign_lines_keeps_real_lines():
    kept, suppressed = _partition_benign_lines(_MIXED_STDERR)
    joined = "\n".join(kept)
    assert "out of memory" in joined.lower()
    assert "modeling_cohere2.py" not in joined
    assert suppressed and "modeling_cohere2.py" in suppressed[0]


def test_classify_root_cause_type():
    assert _classify_root_cause_type("server_init_dead", "HIP out of memory") == "oom"
    assert _classify_root_cause_type("timeout", "baseline benchmark exceeded 600s") == "benchmark_timeout"
    assert _classify_root_cause_type("server_init_dead", "EngineCore failed to start") == "engine_core_init"
    assert _classify_root_cause_type("subprocess_nonzero", "boom") == "worker_crash"
    assert _classify_root_cause_type("unknown", "") == "unknown"


# ---------------------------------------------------------------------------
# failure_summary sourced from SharedState
# ---------------------------------------------------------------------------
def test_failure_summary_from_server_init_dead_attempt():
    state = _failed_state(error_class="server_init_dead", error=_SERVER_LOG_OOM)
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert fs["last_attempt_id"] == "baseline-1"


def test_failure_summary_mixed_error_keeps_real_root_cause():
    """A benign WARN mixed with a real fault must NOT wipe the real root cause."""
    state = _failed_state(error_class="subprocess_nonzero", error=_MIXED_STDERR)
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert "modeling_cohere2.py" not in fs["root_cause"]
    assert fs["suppressed_benign"]


def test_failure_summary_pure_benign_falls_back_to_server_log(tmp_path):
    """When the only recorded error is the benign WARN, fall back to the
    attempt's server.log terminal marker."""
    workspace = tmp_path / "runs" / "baseline" / "baseline-1" / "benchmark_vllm_x"
    workspace.mkdir(parents=True)
    (workspace.parent / "server.log").write_text(_SERVER_LOG_OOM, encoding="utf-8")
    state = _failed_state(
        error_class="unknown", error=_BENIGN_WARN, workspace=str(workspace),
    )
    fs = _build_failure_summary(state, tmp_path)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert fs["suppressed_benign"]
    assert fs["server_log"].endswith("server.log")
    # server.log path is rendered session-relative when session_dir is given.
    assert not fs["server_log"].startswith(str(tmp_path))


def test_failure_summary_picks_latest_failed_attempt():
    state = _failed_state(
        error_class="timeout", error="baseline benchmark exceeded 600s",
        task_id="attempt-old",
    )
    # A later, more severe attempt.
    state.record_action_attempt(
        action="baseline", task_id="attempt-new", status="failed",
        decision="no_promote",
        result={"status": "failed", "error_class": "server_init_dead",
                "error": _SERVER_LOG_OOM},
    )
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["last_attempt_id"] == "attempt-new"
    assert fs["root_cause_type"] == "oom"


def test_failure_summary_falls_back_to_last_action_failures():
    """When no baseline audit row exists, use the global failure log row."""
    state = SharedState()
    state.set_stop_reason("baseline_failed")
    state.record_action_failure(
        action="baseline", task_id="b1",
        result={"status": "failed", "error_class": "server_init_dead",
                "error": _SERVER_LOG_OOM},
    )
    assert not state.baseline_attempts
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert fs["last_attempt_id"] == "b1"


def test_failure_summary_absent_when_not_baseline_failed():
    state = _failed_state(
        error_class="server_init_dead", error=_SERVER_LOG_OOM,
        stop_reason="time_exhausted",
    )
    assert _build_failure_summary(state) is None
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert "failure_summary" not in summary


def test_failure_summary_absent_when_no_failed_attempt():
    state = SharedState()
    state.set_stop_reason("baseline_failed")
    assert _build_failure_summary(state) is None


# ---------------------------------------------------------------------------
# wiring through _build_summary_dict / _format_md
# ---------------------------------------------------------------------------
def test_build_summary_dict_attaches_failure_summary():
    state = _failed_state(error_class="server_init_dead", error=_SERVER_LOG_OOM)
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert summary["failure_summary"]["root_cause_type"] == "oom"


def test_format_md_surfaces_root_cause():
    state = _failed_state(error_class="server_init_dead", error=_SERVER_LOG_OOM)
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    md = _format_md(summary)
    assert "Root cause" in md
    assert "oom" in md
    assert "modeling_cohere2.py" not in md
