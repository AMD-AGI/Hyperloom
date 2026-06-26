# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P7: roofline profile retry falls back to eager on a cuda-graph capture crash.

sglang's torch profiler collides with HIP CUDA-graph stream capture
(hipErrorStreamCaptureUnsupported). The roofline retry must then boot the
profile server eager (--disable-cuda-graph), mirroring the baseline fallback.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    make_roofline_executor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


def _ctx(tmp_path: Path, params: dict | None = None) -> RunnerContext:
    task = Task(
        task_id="t-roofline-cg", kind="roofline", state="running",
        params=params if params is not None
        else {"base_extra_args": "--mem-fraction-static=0.92"},
        idempotency_key="roofline:t-cg",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


def _state(framework: str = "") -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    if framework:
        s.framework = framework
    return s


def _profile_success(trace_path: str = "/tmp/trace.json.gz") -> dict:
    return {"status": "succeeded", "main_trace_path": trace_path,
            "workspace": "/tmp/workspace", "output_throughput": 110.0}


def _ta_success() -> dict:
    return {"status": "ok", "candidates_path": "/tmp/kc.json",
            "trace_report_path": "/tmp/analysis.md", "hot_kernels": [],
            "trace_health_warnings": []}


async def _run(tmp_path, first_result, *, state=None, params=None):
    seen: list[dict] = []
    calls = {"n": 0}

    async def fake_profile(c):
        seen.append(dict(c.task.params or {}))
        calls["n"] += 1
        return first_result if calls["n"] == 1 else _profile_success()

    async def fake_ta(payload, *, session_dir):
        return _ta_success()

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        await make_roofline_executor(shared_state=state or _state())(
            _ctx(tmp_path, params=params),
        )
    return seen


@pytest.mark.asyncio
async def test_cuda_graph_capture_failure_triggers_eager_retry(tmp_path):
    capture_err = {
        "status": "failed", "error_class": "server_init_dead",
        "error": (
            "Capture cuda graph failed: HIP error: operation not permitted "
            "when stream is capturing (hipErrorStreamCaptureUnsupported)"
        ),
    }
    seen = await _run(tmp_path, capture_err)
    assert len(seen) >= 2
    assert "--disable-cuda-graph" not in str(seen[0].get("base_extra_args", ""))
    assert "--disable-cuda-graph" in str(seen[1].get("base_extra_args", ""))


@pytest.mark.asyncio
async def test_non_capture_failure_does_not_force_eager(tmp_path):
    other_err = {"status": "failed", "error_class": "timeout",
                 "error": "Read timed out. (read timeout=600)"}
    seen = await _run(tmp_path, other_err)
    assert len(seen) >= 2
    assert "--disable-cuda-graph" not in str(seen[1].get("base_extra_args", ""))


_CAPTURE_ERR = {
    "status": "failed", "error_class": "server_init_dead",
    "error": (
        "Capture cuda graph failed: HIP error: operation not permitted "
        "when stream is capturing (hipErrorStreamCaptureUnsupported)"
    ),
}


@pytest.mark.asyncio
async def test_vllm_eager_retry_uses_enforce_eager_from_shared_state(tmp_path, caplog):
    # vLLM rejects sglang's --disable-cuda-graph; the framework comes from
    # shared_state (the internal roofline task params omit it), so the retry
    # must inject --enforce-eager.
    seen = await _run(
        tmp_path, _CAPTURE_ERR, state=_state(framework="vllm"),
        params={"base_extra_args": "--gpu-memory-utilization=0.9"},
    )
    assert len(seen) >= 2
    retry_args = str(seen[1].get("base_extra_args", ""))
    assert "--enforce-eager" in retry_args
    assert "--disable-cuda-graph" not in retry_args
    assert "next attempt boots eager (--enforce-eager)" in caplog.text
    assert "next attempt boots eager (--disable-cuda-graph)" not in caplog.text


@pytest.mark.asyncio
async def test_vllm_eager_retry_uses_framework_env(tmp_path, monkeypatch):
    # When neither params nor shared_state carries it, FRAMEWORK env wins.
    monkeypatch.setenv("FRAMEWORK", "vllm")
    seen = await _run(
        tmp_path, _CAPTURE_ERR,
        params={"base_extra_args": "--gpu-memory-utilization=0.9"},
    )
    assert len(seen) >= 2
    retry_args = str(seen[1].get("base_extra_args", ""))
    assert "--enforce-eager" in retry_args
    assert "--disable-cuda-graph" not in retry_args


# #735: a profile that produced only CUDA-graph capture sidecars (capture-only
# fallback) carries no per-iteration annotations; the steady-state splitter
# would die downstream with the misleading trace_split_no_steady_state. The
# roofline retry must treat this as transient: re-profile (escalating to eager),
# and only fail with an accurate message if every attempt stays capture-only.
_CAPTURE_ONLY = {
    "status": "succeeded",
    "main_trace_path": "/tmp/ws/torch_trace",
    "trace_files": ["/tmp/ws/torch_trace/capture_traces/bs_104_rank0.json.gz"],
    "workspace": "/tmp/ws",
    "profile_trace_selection_reason": "capture_only_fallback",
}


@pytest.mark.asyncio
async def test_capture_only_profile_retries_then_succeeds(tmp_path):
    # First attempt is capture-only -> retry escalates to eager and the second
    # attempt produces a real annotated trace, so the run proceeds.
    seen = await _run(tmp_path, _CAPTURE_ONLY)
    assert len(seen) >= 2
    # The retry boots eager so the steady-state window gets annotated.
    assert "--disable-cuda-graph" in str(seen[1].get("base_extra_args", ""))


@pytest.mark.asyncio
async def test_capture_only_every_attempt_fails_clearly(tmp_path):
    # Every attempt stays capture-only -> terminal failure with an accurate
    # phase, NOT the misleading downstream trace_split_no_steady_state.
    async def always_capture_only(c):
        return dict(_CAPTURE_ONLY)

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=always_capture_only,
    ):
        result = await make_roofline_executor(shared_state=_state())(_ctx(tmp_path))
    assert result["status"] == "failed"
    assert result.get("phase") == "profile_capture_only"
    assert "capture" in str(result.get("error", "")).lower()
