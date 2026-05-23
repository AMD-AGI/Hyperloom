"""N26 — RooflineExecutor auto-retries trace_analyze on
steady_state_chunk_{empty,missing} with TraceLens-supplied alternate mode.

Background (continues from N25):

N25 made the TraceLens splitter's chunk selection explicit
(``--steady-state-mode`` flag + ``INFERENCE_OPTIMIZER_STEADY_STATE_MODE``
env) and hard-fails when the requested chunk is structurally empty
(``num_gpu_events==0`` or ``gpu_busy_duration==0.0``). The failure
warning carries ``non_empty_modes`` -- the alternate modes whose
chunks DO have GPU events (per splitter's own
``execution_details.csv``).

Pre-N26 this meant operators had to manually:
  1. read the warning
  2. kill the session
  3. ``export INFERENCE_OPTIMIZER_STEADY_STATE_MODE=<other_mode>``
  4. restart

N26 closes that loop: RooflineExecutor reads the warning's
``non_empty_modes``, re-issues trace_analyze ONCE with the first
alternate, and proceeds normally if that succeeds. This is NOT an
inference_optimizer-side heuristic -- the alternate comes straight
from TraceLens splitter's structured recovery hint. We are simply
consuming what TraceLens already tells us.

Tests pinned here:

* Happy path unchanged -- N26 only activates on the specific failure.
* ``steady_state_chunk_empty`` + ``non_empty_modes=['prefilldecode']``
  -> auto-retry with steady_state_mode=prefilldecode succeeds, executor
  returns succeeded.
* ``steady_state_chunk_missing`` + ``available_modes=[...]`` -> same
  auto-retry behaviour.
* Empty ``non_empty_modes`` (splitter says no alternate has work) ->
  NO retry, propagate original failure.
* Unknown failure code -> NO retry, propagate original failure.
* First retry attempt ALSO fails -> propagate the retry's failure,
  don't loop a second time.
* Retry succeeds -> result carries ``n26_auto_retry`` field with
  ``from_mode`` + ``to_mode`` + ``source_warning_code``.
* Single-retry invariant: when the initial call already used a
  non-mixed mode (e.g. operator passed prefilldecode explicitly and
  it still came back empty), we DO retry once if a different
  non_empty_mode exists.
* SharedState ``last_trace_analyze`` is correctly populated on retry
  success (same C1 path as happy path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    RooflineExecutor,
    _extract_steady_state_retry_mode,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ctx(tmp_path: Path) -> RunnerContext:
    task = Task(
        task_id="t-n26-1", kind="roofline", state="running",
        params={"base_extra_args": ""},
        idempotency_key="roofline:n26-1",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


def _state() -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    return s


def _profile_ok(trace: str = "/tmp/t.gz") -> dict:
    return {
        "status": "succeeded",
        "main_trace_path": trace,
        "workspace": "/tmp/wsp",
        "output_throughput": 50.0,
    }


def _ta_empty_chunk_failure(
    *, requested: str = "mixed", non_empty: list[str] | None = None,
) -> dict:
    """trace_analyze_handler result shape after N25 hard-fail on empty
    selected chunk."""
    return {
        "status": "failed",
        "error": (
            f"RuntimeError: steady_state_chunk_empty: requested "
            f"--steady-state-mode={requested} but the selected chunk has "
            "zero GPU events"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_empty",
                "severity": "blocking",
                "requested_mode": requested,
                "selected_chunk": f"/tmp/{requested}.json.gz",
                "num_gpu_events": 0,
                "gpu_busy_duration": 0.0,
                "non_empty_modes": list(non_empty if non_empty is not None else []),
                "remediation": "Re-issue with one of non_empty_modes.",
                "message": "stub",
            },
        ],
    }


def _ta_missing_chunk_failure(
    *, requested: str = "decode_only", available: list[str] | None = None,
) -> dict:
    return {
        "status": "failed",
        "error": (
            f"RuntimeError: steady_state_chunk_missing: requested "
            f"--steady-state-mode={requested} but splitter produced no "
            "matching chunk"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_missing",
                "severity": "blocking",
                "requested_mode": requested,
                "requested_chunk_label": f"{requested}_steady_state",
                "available_modes": list(available if available is not None else []),
                "remediation": "Re-issue with one of available_modes.",
                "trace_input": "/tmp/raw.trace.json.gz",
                "split_dir": "/tmp/split",
            },
        ],
    }


def _ta_ok(*, report_md: Path) -> dict:
    return {
        "status": "ok",
        "candidates_path": "/tmp/kc.json",
        "trace_report_path": str(report_md),
        "hot_kernels": [],
        "trace_health_warnings": [],
    }


# ---------------------------------------------------------------------------
# _extract_steady_state_retry_mode pure-function contract
# ---------------------------------------------------------------------------


def test_extract_picks_first_non_empty_mode():
    res = _ta_empty_chunk_failure(non_empty=["prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    mode, w = out
    assert mode == "prefilldecode"
    assert w["code"] == "steady_state_chunk_empty"


def test_extract_handles_missing_chunk_warning():
    res = _ta_missing_chunk_failure(available=["mixed", "prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    mode, w = out
    assert mode == "mixed"
    assert w["code"] == "steady_state_chunk_missing"


def test_extract_returns_none_when_no_alternate():
    """non_empty_modes=[] -> splitter has no usable chunk anywhere ->
    we don't synthesize a retry mode."""
    res = _ta_empty_chunk_failure(non_empty=[])
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_returns_none_for_unrelated_warning():
    res = {
        "status": "failed",
        "trace_health_warnings": [
            {"code": "tracelens_analysis_failed", "severity": "warning"},
        ],
    }
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_returns_none_for_empty_result():
    assert _extract_steady_state_retry_mode({}) is None
    assert _extract_steady_state_retry_mode({"status": "ok"}) is None
    assert _extract_steady_state_retry_mode(  # type: ignore[arg-type]
        None
    ) is None


def test_extract_skips_blank_mode_entries():
    """Defensive: a malformed warning with empty-string modes shouldn't
    be selected."""
    res = _ta_empty_chunk_failure(non_empty=["", "   ", "prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    assert out[0] == "prefilldecode"


# ---------------------------------------------------------------------------
# RooflineExecutor end-to-end behaviour
# ---------------------------------------------------------------------------


def _patch_subs(profile_result, ta_results):
    """Sequence of trace_analyze results -- one per call. Use this to
    simulate "first call fails, second call (auto-retry) succeeds"."""
    ta_calls = {"n": 0, "payloads": []}

    async def fake_profile(ctx):
        return profile_result

    async def fake_ta(payload, *, session_dir):
        idx = ta_calls["n"]
        ta_calls["payloads"].append(dict(payload))
        ta_calls["n"] += 1
        if idx >= len(ta_results):
            return ta_results[-1]
        return ta_results[idx]

    return (
        patch(
            "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
        ta_calls,
    )


@pytest.mark.asyncio
async def test_auto_retry_succeeds_on_alternate_mode(tmp_path):
    """SOLAR-style: mixed fails with non_empty_modes=['prefilldecode'],
    N26 auto-retries with steady_state_mode=prefilldecode, retry
    succeeds, RooflineExecutor returns succeeded normally."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="mixed", non_empty=["prefilldecode"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert result["snapshot_id"] == 1
    # Two trace_analyze calls were made; the second carried the retry mode.
    assert calls["n"] == 2
    assert calls["payloads"][0].get("steady_state_mode") is None
    assert calls["payloads"][1]["steady_state_mode"] == "prefilldecode"
    assert calls["payloads"][1].get("_n26_auto_retry") is True
    assert calls["payloads"][1].get("_n26_retry_from_mode") == "mixed"
    # SharedState carries the recovered snapshot.
    assert state.last_profile_trace == "/tmp/t.gz"
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(md)
    assert cached["roofline_snapshot_id"] == 1


@pytest.mark.asyncio
async def test_auto_retry_succeeds_on_missing_chunk(tmp_path):
    """steady_state_chunk_missing path: requested mode not produced;
    available_modes lists what splitter did make."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 80%, Idle 20%\n", encoding="utf-8")
    fail = _ta_missing_chunk_failure(
        requested="decode_only", available=["mixed", "prefilldecode"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["payloads"][1]["steady_state_mode"] == "mixed"
    assert calls["payloads"][1]["_n26_retry_from_mode"] == "decode_only"


@pytest.mark.asyncio
async def test_no_retry_when_no_alternate_modes(tmp_path):
    """non_empty_modes=[] -> splitter has nothing for us; bubble up the
    original failure unchanged."""
    fail = _ta_empty_chunk_failure(requested="mixed", non_empty=[])
    p1, p2, calls = _patch_subs(_profile_ok(), [fail])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    # `phase` (set by _failed) names the sub-step; `error` is the
    # upstream message (steady_state_chunk_empty). Both surfaces matter.
    assert result.get("phase") == "trace_analyze"
    assert "steady_state_chunk_empty" in str(result.get("error") or "")
    assert calls["n"] == 1  # ONE call total -- no retry.
    assert state.last_trace_analyze == {}


@pytest.mark.asyncio
async def test_no_retry_on_unrelated_failure(tmp_path):
    """tracelens_analysis_failed (and other codes) don't trigger N26 --
    only steady_state_chunk_{empty,missing} do."""
    fail = {
        "status": "failed",
        "error": "RuntimeError: TraceLens skill crashed",
        "trace_health_warnings": [
            {"code": "tracelens_analysis_failed", "severity": "warning"},
        ],
    }
    p1, p2, calls = _patch_subs(_profile_ok(), [fail])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_failure_propagates_without_third_attempt(tmp_path):
    """If the auto-retry ALSO fails, surface the retry's failure --
    never escalate to a third attempt."""
    fail1 = _ta_empty_chunk_failure(
        requested="mixed", non_empty=["prefilldecode"],
    )
    fail2 = {
        "status": "failed",
        "error": (
            "RuntimeError: steady_state_chunk_empty: requested "
            "--steady-state-mode=prefilldecode but the selected chunk has "
            "zero GPU events"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_empty",
                "requested_mode": "prefilldecode",
                "non_empty_modes": ["decode_only"],
            },
        ],
    }
    p1, p2, calls = _patch_subs(_profile_ok(), [fail1, fail2])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    assert calls["n"] == 2  # exactly TWO calls -- no third retry.
    # The retry's failure error is what surfaces (it contains
    # prefilldecode, not the original mixed).
    err = str(result.get("error") or "")
    assert "prefilldecode" in err


@pytest.mark.asyncio
async def test_retry_exception_propagates(tmp_path):
    """If the retry trace_analyze raises (network blip, etc.), we
    surface a clear failed result mentioning N26 + the chosen mode."""
    fail = _ta_empty_chunk_failure(
        requested="mixed", non_empty=["prefilldecode"],
    )

    async def fake_profile(ctx):
        return _profile_ok()

    call_count = {"n": 0}

    async def fake_ta(payload, *, session_dir):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return fail
        raise RuntimeError("network unreachable")

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    err = str(result.get("error") or "")
    assert "N26 auto-retry" in err
    assert "prefilldecode" in err


@pytest.mark.asyncio
async def test_retry_success_stamps_n26_metadata(tmp_path):
    """The retry-success result must carry n26_auto_retry metadata so
    the recorder / prompt renderer can surface it to the LLM."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="mixed", non_empty=["prefilldecode"],
    )
    succ = _ta_ok(report_md=md)

    # We need to capture what the executor passes into record_trace_analyze.
    captured = {"ta_result": None}

    async def fake_profile(ctx):
        return _profile_ok()

    async def fake_ta(payload, *, session_dir):
        return fail if not payload.get("_n26_auto_retry") else succ

    state = _state()

    orig_record = state.record_trace_analyze

    def record_spy(payload, result):
        captured["ta_result"] = dict(result)
        return orig_record(payload, result)

    state.record_trace_analyze = record_spy  # type: ignore[assignment]

    executor = RooflineExecutor(shared_state=state)
    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    stamp = (captured["ta_result"] or {}).get("n26_auto_retry") or {}
    assert stamp.get("applied") is True
    assert stamp.get("from_mode") == "mixed"
    assert stamp.get("to_mode") == "prefilldecode"
    assert stamp.get("source_warning_code") == "steady_state_chunk_empty"


@pytest.mark.asyncio
async def test_retry_works_when_operator_started_with_non_mixed(tmp_path):
    """Single-retry invariant: when operator already set
    INFERENCE_OPTIMIZER_STEADY_STATE_MODE=prefilldecode and even that
    came back empty, N26 still tries once more with the next
    non_empty mode (e.g. decode_only) -- the cap is ONE retry per
    RooflineExecutor invocation, not "no retry when not from mixed"."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="prefilldecode", non_empty=["decode_only"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["payloads"][1]["steady_state_mode"] == "decode_only"
    assert calls["payloads"][1]["_n26_retry_from_mode"] == "prefilldecode"
