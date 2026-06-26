# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline retry boots eager on sglang's profile-cuda-graph seq_lens assert.

sglang's profile-cuda-graph shape discovery feeds a device ``seq_lens`` into
``get_num_new_pages`` (which asserts CPU) -> AssertionError -> SIGQUIT. The crash
surfaces only in the engine ``server.log``, and a bare AssertionError is otherwise
treated as non-recoverable. The roofline retry must still recognise this specific
assert and boot the next attempt eager (--disable-cuda-graph).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.baseline import (
    _is_cuda_graph_capture_failure,
)
from inference_optimizer.orchestrator.action_executors.roofline import (
    make_roofline_executor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task

# Real crash tail captured from a failing roofline run (issue #622, Part A).
_SEQ_LENS_ASSERT_LOG = """\
[2026-06-16 21:18:45] INFO:     Application startup complete.
[2026-06-16 21:18:46] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 2765, in _get_new_batch_prefill_raw
    new_batch.prepare_for_extend()
  File "/sgl-workspace/sglang/python/sglang/srt/mem_cache/common.py", line 467, in alloc_for_extend
    out_cache_loc = alloc_paged_token_slots_extend(
  File "/sgl-workspace/sglang/python/sglang/srt/mem_cache/allocator.py", line 442, in alloc_extend
    num_new_pages = get_num_new_pages(
  File "/sgl-workspace/sglang/python/sglang/srt/utils/common.py", line 3767, in get_num_new_pages
    assert seq_lens.device == cpu_device
AssertionError

[2026-06-16 21:18:46] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
"""


def test_detector_recognises_seq_lens_assert_as_recoverable():
    # Bug repro: this exact assert returned False before the fix (assertionerror
    # was a blanket non-recoverable marker).
    assert _is_cuda_graph_capture_failure(_SEQ_LENS_ASSERT_LOG) is True


def test_detector_keeps_generic_assertion_non_recoverable():
    # Guard: a generic AssertionError without the seq_lens markers stays
    # non-recoverable, so we don't waste the eager retry.
    generic = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in f\n    assert cond\nAssertionError\n'
    )
    assert _is_cuda_graph_capture_failure(generic) is False


def _ctx(tmp_path: Path) -> RunnerContext:
    task = Task(
        task_id="t-roofline-seqlens", kind="roofline", state="running",
        params={"base_extra_args": "--mem-fraction-static=0.92"},
        idempotency_key="roofline:t-seqlens",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


def _state() -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    return s


def _profile_success() -> dict:
    return {"status": "succeeded", "main_trace_path": "/tmp/trace.json.gz",
            "workspace": "/tmp/workspace", "output_throughput": 110.0}


def _ta_success() -> dict:
    return {"status": "ok", "candidates_path": "/tmp/kc.json",
            "trace_report_path": "/tmp/analysis.md", "hot_kernels": [],
            "trace_health_warnings": []}


@pytest.mark.asyncio
async def test_seq_lens_assert_in_server_log_triggers_eager_retry(tmp_path):
    # The crash lives only in server.log; the failed result merely points to its
    # trace_dir. The retry must read that log and boot eager.
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "server.log").write_text(_SEQ_LENS_ASSERT_LOG, encoding="utf-8")
    first_result = {
        "status": "failed", "error_class": "server_init_dead",
        "error": "profile sub-step failed",  # no seq_lens hint here
        "trace_dir": str(trace_dir),
    }

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
        await make_roofline_executor(shared_state=_state())(_ctx(tmp_path))

    assert len(seen) >= 2
    assert "--disable-cuda-graph" not in str(seen[0].get("base_extra_args", ""))
    assert "--disable-cuda-graph" in str(seen[1].get("base_extra_args", ""))
