# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``baseline_failed`` root-cause surfacing tests."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors import report
from hyperloom.orchestrator.actions.executors.report import (
    _build_failure_summary,
    _build_summary_dict,
    _format_md,
    _highlight_is_benign,
    _is_benign_failure_text,
    _partition_benign_lines,
)
from hyperloom.orchestrator.state.failure_evidence import classify_root_cause_type
from hyperloom.orchestrator.state.shared_state import ActionAttempt, SharedState

_BENIGN_WARN = (
    "[WARN] Error: [Errno 2] No such file or directory: '/app/.../transformers/models/cohere2/modeling_cohere2.py'"
)

_SERVER_LOG_OOM = (
    "non-default args: {'tensor_parallel_size': 8}\n"
    "[aiter] import [module_aiter_core] ...\n"
    "[2] [FATAL ERROR]: HIP failure: 'out of memory' "
    "(in ncclCommInitRank during init_model_parallel)\n"
    "RuntimeError: NCCL error: unhandled cuda error\n"
    "EngineCore failed to start.\n"
)

# A subprocess_nonzero stderr tail that mixes the benign WARN with the real fault on separate lines.
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
    """Build a SharedState carrying one failed baseline attempt."""
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
        action="baseline", attempt=ActionAttempt(task_id=task_id, status="failed", decision="no_promote", result=result)
    )
    state.record_action_failure(
        action="baseline",
        task_id=task_id,
        result=result,
    )
    return state


def test_is_benign_failure_text():
    assert _is_benign_failure_text(_BENIGN_WARN)
    assert not _is_benign_failure_text("HIP out of memory during ncclCommInitRank")


def test_highlight_is_benign_uses_summary_only():
    # Benign headline → suppressed.
    assert _highlight_is_benign({"summary": _BENIGN_WARN, "payload": {}})
    # Real fault headline is kept even when payload mentions the benign file.
    assert not _highlight_is_benign({"summary": "sev=high EngineCore failed", "payload": {"f": _BENIGN_WARN}})


def test_partition_benign_lines_keeps_real_lines():
    kept, suppressed = _partition_benign_lines(_MIXED_STDERR)
    joined = "\n".join(kept)
    assert "out of memory" in joined.lower()
    assert "modeling_cohere2.py" not in joined
    assert suppressed and "modeling_cohere2.py" in suppressed[0]


def test_classify_root_cause_type():
    assert classify_root_cause_type("server_init_dead", "HIP out of memory") == "oom"
    assert classify_root_cause_type("timeout", "baseline benchmark exceeded 600s") == "benchmark_timeout"
    assert classify_root_cause_type("server_init_dead", "EngineCore failed to start") == "engine_core_init"
    assert classify_root_cause_type("subprocess_nonzero", "boom") == "unknown"
    assert classify_root_cause_type("unknown", "") == "unknown"


def test_classify_root_cause_type_needs_evidence_of_a_death():
    """A nonzero exit Hyperloom could not explain is not a crash on its own."""
    clean_shutdown = (
        "replay finished: 4097/4097 requests served in 4486.0s\n"
        "power validation failed: expected 4 GPUs, sampled 8\n"
        "INFO: server received SIGTERM, worker 3 of 4 shutting down\n"
    )
    assert classify_root_cause_type("subprocess_nonzero", clean_shutdown) == "unknown"
    assert (
        classify_root_cause_type("subprocess_nonzero", "RuntimeError: NCCL error: unhandled system error")
        == "worker_crash"
    )
    assert classify_root_cause_type("subprocess_nonzero", "rank 3 killed by signal 11 (core dumped)") == "worker_crash"


def test_classify_root_cause_type_benign_nccl_warn_is_not_a_crash():
    """NCCL prints WARN lines on healthy launches, so a warning is not evidence of a death."""
    warn = (
        "ip-10-0-0-4:142:367 [0] NCCL WARN Missing ncclnet symbol, using internal network plugin\n"
        "replay finished: 4097/4097 requests served in 4486.0s\n"
    )
    assert classify_root_cause_type("subprocess_nonzero", warn) == "unknown"


def test_classify_root_cause_type_hardware_deaths():
    """The ways a worker dies on this hardware each count as evidence of a death."""
    assert (
        classify_root_cause_type(
            "subprocess_nonzero",
            "Memory access fault by GPU node-2 on address 0x7f1c00000000. Reason: Page not present",
        )
        == "worker_crash"
    )
    assert classify_root_cause_type("subprocess_nonzero", "Fatal Python error: Aborted") == "worker_crash"
    assert (
        classify_root_cause_type("subprocess_nonzero", "CUDA error: an illegal memory access was encountered")
        == "worker_crash"
    )
    assert (
        classify_root_cause_type(
            "subprocess_nonzero", "terminate called after throwing an instance of 'c10::DistBackendError'"
        )
        == "worker_crash"
    )
    # A negative exit status is a signal, however the launcher spaces it out.
    assert classify_root_cause_type("subprocess_nonzero", "  exitcode  : -6 (pid: 118231)") == "worker_crash"
    assert classify_root_cause_type("subprocess_nonzero", "exitcode: -11") == "worker_crash"


def test_classify_root_cause_type_death_outranks_the_harness_class():
    """A crash after the last request is a crash, not the harness refusing the measurement."""
    assert (
        classify_root_cause_type(
            "magpie_nonzero_invalid_measurement", "Memory access fault by GPU node-2 on address 0x7f1c00000000"
        )
        == "worker_crash"
    )


def test_classify_root_cause_type_harness_class():
    """A class the attribution table calls harness-side is reported as such."""
    assert (
        classify_root_cause_type("magpie_nonzero_invalid_measurement", "benchmark exited 3 after the last request")
        == "benchmark_harness"
    )


def test_failure_summary_unexplained_exit_keeps_the_excerpt():
    """Without evidence of a death the excerpt stays the headline, not a diagnosis."""
    state = _failed_state(
        error_class="subprocess_nonzero",
        error="power validation failed: expected 4 GPUs, sampled 8",
    )
    summary = _build_failure_summary(state)
    assert summary["root_cause_type"] == "unknown"
    assert "power validation failed" in summary["root_cause"]


def test_failure_summary_from_server_init_dead_attempt():
    state = _failed_state(error_class="server_init_dead", error=_SERVER_LOG_OOM)
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert fs["last_attempt_id"] == "baseline-1"


def test_failure_summary_build_error_is_not_reported_as_no_failed_attempt(monkeypatch):
    def _broken_headline(text):
        raise TypeError("headline picker broke")

    monkeypatch.setattr(report, "_pick_failure_headline", _broken_headline)
    state = _failed_state(error_class="server_init_dead", error=_SERVER_LOG_OOM)
    with pytest.raises(TypeError, match="headline picker broke"):
        _build_failure_summary(state)


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
    """When the only recorded error is the benign WARN, fall back to the attempt's server.log terminal marker."""
    workspace = tmp_path / "runs" / "baseline" / "baseline-1" / "benchmark_vllm_x"
    workspace.mkdir(parents=True)
    (workspace.parent / "server.log").write_text(_SERVER_LOG_OOM, encoding="utf-8")
    state = _failed_state(
        error_class="unknown",
        error=_BENIGN_WARN,
        workspace=str(workspace),
    )
    fs = _build_failure_summary(state, tmp_path)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert "out of memory" in fs["root_cause"].lower()
    assert fs["suppressed_benign"]
    assert fs["server_log"].endswith("server.log")
    # server.log path is session-relative when session_dir is given.
    assert not fs["server_log"].startswith(str(tmp_path))


def test_failure_summary_picks_latest_failed_attempt():
    state = _failed_state(
        error_class="timeout",
        error="baseline benchmark exceeded 600s",
        task_id="attempt-old",
    )
    # A later, more severe attempt.
    state.record_action_attempt(
        action="baseline",
        attempt=ActionAttempt(
            task_id="attempt-new",
            status="failed",
            decision="no_promote",
            result={"status": "failed", "error_class": "server_init_dead", "error": _SERVER_LOG_OOM},
        ),
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
        action="baseline",
        task_id="b1",
        result={"status": "failed", "error_class": "server_init_dead", "error": _SERVER_LOG_OOM},
    )
    assert not state.baseline_attempts
    fs = _build_failure_summary(state)
    assert fs is not None
    assert fs["root_cause_type"] == "oom"
    assert fs["last_attempt_id"] == "b1"


def test_failure_summary_absent_when_not_baseline_failed():
    state = _failed_state(
        error_class="server_init_dead",
        error=_SERVER_LOG_OOM,
        stop_reason="time_exhausted",
    )
    assert _build_failure_summary(state) is None
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert "failure_summary" not in summary


def test_failure_summary_absent_when_no_failed_attempt():
    state = SharedState()
    state.set_stop_reason("baseline_failed")
    assert _build_failure_summary(state) is None


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


def test_classify_root_cause_type_kv_cache_oom():
    assert (
        classify_root_cause_type(
            "kv_cache_oom",
            "Loaded weights leave no GPU memory for the KV cache",
        )
        == "kv_cache_oom"
    )
    assert (
        classify_root_cause_type(
            "subprocess_nonzero",
            "Raise --mem-fraction-static above 0.737",
        )
        == "kv_cache_oom"
    )


def test_classify_root_cause_ignores_mem_fraction_static_in_argv():
    # A generic failure whose text merely echoes --mem-fraction-static in the launch command must not be mislabeled
    # kv_cache_oom.
    result = classify_root_cause_type(
        "subprocess_nonzero",
        "server cmd: python -m sglang.launch_server --mem-fraction-static 0.9 --tp 8; exited 1 with a segfault",
    )
    assert result != "kv_cache_oom"


def test_classify_root_cause_kv_cache_oom_specific_phrase_still_matches():
    # The "Raise --mem-fraction-static above" remediation line is a genuine KV-cache OOM signal even when error_class
    # is generic.
    assert (
        classify_root_cause_type(
            "subprocess_nonzero",
            "ValueError: ... Raise --mem-fraction-static above 0.737",
        )
        == "kv_cache_oom"
    )
