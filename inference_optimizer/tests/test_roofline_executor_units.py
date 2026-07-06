# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

"""Focused unit tests for roofline executor guard/branch/fallback paths.

These target the residual uncovered lines in
``src/hyperloom/orchestrator/action_executors/roofline.py``:
non-dict / bad-shape guards in the pure helpers, the exception-path
cuda-graph escalation, the ``close_post_opt`` output-name branch, the
retry-returns-non-dict path, the session-dir save fast-paths (success +
defensive except), and the ``_resolve_framework`` params fast-path.

All tests are hermetic: profile / trace_analyze boundaries are stubbed,
filesystem uses ``tmp_path`` only, no GPU / subprocess / network.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.action_executors import roofline as rf
from hyperloom.orchestrator.action_executors.roofline import (
    RooflineExecutor,
    _extract_steady_state_retry_mode,
    _profile_err_text,
    _profile_server_log_tail,
    make_roofline_executor,
)
from hyperloom.orchestrator.shared_state import SharedState
from hyperloom.orchestrator.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.task_registry import Task


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _state() -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    return s


def _ctx(tmp_path: Path | None = None, *, params: dict | None = None) -> RunnerContext:
    task = Task(
        task_id="t-rf-units",
        kind="roofline",
        state="running",
        params=params if params is not None else {"base_extra_args": ""},
        idempotency_key="roofline:units-1",
        requires_lanes=["profile_lane"],
    )
    extra: dict = {}
    if tmp_path is not None:
        extra["session_dir"] = str(tmp_path)
    return RunnerContext(task=task, lease=None, extra=extra)


def _profile_success(trace_path: str = "/tmp/trace.json.gz") -> dict:
    return {
        "status": "succeeded",
        "main_trace_path": trace_path,
        "workspace": "/tmp/workspace",
        "output_throughput": 110.0,
    }


def _ta_ok(report_md: Path) -> dict:
    return {
        "status": "ok",
        "candidates_path": "/tmp/kc.json",
        "trace_report_path": str(report_md),
        "hot_kernels": [],
        "trace_health_warnings": [],
    }


def _patch_subs(profile_result, ta_result):
    async def fake_profile(ctx):
        if isinstance(profile_result, Exception):
            raise profile_result
        return profile_result

    async def fake_ta(payload, *, session_dir):
        if isinstance(ta_result, Exception):
            raise ta_result
        return ta_result

    return patch(
        "hyperloom.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    )


def _seed_session_dir(session_dir: Path, state: SharedState) -> None:
    """Materialize a real state.json so the save() fast-paths execute."""
    session_dir.mkdir(parents=True, exist_ok=True)
    state.save(session_dir)
    assert (session_dir / "state.json").exists()


# --------------------------------------------------------------------------
# _extract_steady_state_retry_mode guards (lines 75, 78, 85)
# --------------------------------------------------------------------------
def test_extract_returns_none_when_warnings_not_a_list():
    # line 75: trace_health_warnings present but not a list.
    res = {"status": "failed", "trace_health_warnings": {"code": "x"}}
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_skips_non_dict_warning_entries():
    # line 78: a non-dict entry in the warnings list is skipped; a valid
    # entry after it is still honoured.
    res = {
        "status": "failed",
        "trace_health_warnings": [
            "not-a-dict",
            123,
            {
                "code": "steady_state_chunk_empty",
                "non_empty_modes": ["prefilldecode"],
            },
        ],
    }
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    assert out[0] == "prefilldecode"


def test_extract_skips_warning_when_modes_not_a_list():
    # line 85: the recovery warning names an alternate field that is NOT a
    # list -> skip it; a later well-formed warning still matches.
    res = {
        "status": "failed",
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_missing",
                "non_empty_modes": "prefilldecode",  # str, not a list
            },
            {
                "code": "steady_state_chunk_low_quality",
                "available_modes": ["decode_only"],
            },
        ],
    }
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    assert out[0] == "decode_only"
    assert out[1]["code"] == "steady_state_chunk_low_quality"


# --------------------------------------------------------------------------
# _profile_err_text (lines 163, 170)
# --------------------------------------------------------------------------
def test_profile_err_text_non_dict_returns_empty():
    # line 163
    assert _profile_err_text(None) == ""
    assert _profile_err_text("garbage") == ""
    assert _profile_err_text(42) == ""


def test_profile_err_text_includes_sub_result_fields():
    # line 170: a dict sub_result contributes its error / error_class.
    blob = _profile_err_text(
        {
            "error": "top-level err",
            "error_class": "server_init_dead",
            "sub_result": {
                "error": "nested seq_lens assert",
                "error_class": "nested_class",
            },
        }
    )
    assert "top-level err" in blob
    assert "server_init_dead" in blob
    assert "nested seq_lens assert" in blob
    assert "nested_class" in blob


def test_profile_err_text_ignores_non_dict_sub_result():
    blob = _profile_err_text({"error": "e", "sub_result": "not-a-dict"})
    assert "e" in blob
    assert "not-a-dict" not in blob


# --------------------------------------------------------------------------
# _profile_server_log_tail (lines 182, 193-194)
# --------------------------------------------------------------------------
def test_profile_server_log_tail_non_dict_returns_empty():
    # line 182
    assert _profile_server_log_tail(None) == ""
    assert _profile_server_log_tail("garbage") == ""


def test_profile_server_log_tail_no_base_returns_empty():
    assert _profile_server_log_tail({}) == ""
    assert _profile_server_log_tail({"trace_dir": ""}) == ""


def test_profile_server_log_tail_reads_newest_log(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "server.log").write_text("hello\nSIGQUIT received\n", encoding="utf-8")
    out = _profile_server_log_tail({"trace_dir": str(trace_dir)})
    assert "SIGQUIT received" in out


def test_profile_server_log_tail_swallows_oserror(monkeypatch, tmp_path):
    # lines 193-194: _find_server_logs succeeds but read_bytes raises OSError
    # -> best-effort "" instead of propagating.
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    log = trace_dir / "server.log"
    log.write_text("data", encoding="utf-8")

    real_read_bytes = Path.read_bytes

    def boom(self):
        if self.name == "server.log":
            raise OSError("simulated read failure")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    assert _profile_server_log_tail({"trace_dir": str(trace_dir)}) == ""


def test_profile_server_log_tail_swallows_importerror(monkeypatch):
    # lines 193-194 via ImportError branch: patch the lazy import target so
    # importing _find_server_logs raises.
    import hyperloom.orchestrator.action_executors.benchmark_result as br

    def boom(_slot):
        raise ImportError("simulated import failure")

    monkeypatch.setattr(br, "_find_server_logs", boom)
    assert _profile_server_log_tail({"workspace": "/nonexistent"}) == ""


def test_profile_server_log_tail_empty_when_no_logs(tmp_path):
    # base resolves but _find_server_logs returns [] -> "".
    empty = tmp_path / "empty_ws"
    empty.mkdir()
    assert _profile_server_log_tail({"workspace": str(empty)}) == ""


# --------------------------------------------------------------------------
# Exception-path cuda-graph escalation (lines 311-312)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_profile_exception_with_capture_signature_escalates_eager(tmp_path):
    """profile_executor RAISES an exception whose repr carries the cuda-graph
    capture signature -> next attempt boots eager (lines 308-312)."""
    seen: list[dict] = []
    calls = {"n": 0}

    capture_exc = RuntimeError(
        "Capture cuda graph failed: HIP error: operation not permitted "
        "when stream is capturing (hipErrorStreamCaptureUnsupported)"
    )

    async def fake_profile(ctx):
        seen.append(dict(ctx.task.params or {}))
        calls["n"] += 1
        if calls["n"] == 1:
            raise capture_exc
        return _profile_success()

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    async def fake_ta(payload, *, session_dir):
        return _ta_ok(md)

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with patch(
        "hyperloom.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert len(seen) >= 2
    assert "--disable-cuda-graph" not in str(seen[0].get("base_extra_args", ""))
    assert "--disable-cuda-graph" in str(seen[1].get("base_extra_args", ""))


# --------------------------------------------------------------------------
# close_post_opt output-name branch (line 446)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_close_post_opt_reason_uses_opt_output_name(tmp_path):
    """reason=close_post_opt routes to kernel_roofline_opt.json (line 446)."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    captured: dict = {}

    async def fake_profile(ctx):
        return _profile_success("/tmp/t.gz")

    async def fake_ta(payload, *, session_dir):
        captured["payload"] = dict(payload)
        return _ta_ok(md)

    state = _state()
    ctx = _ctx(tmp_path, params={"base_extra_args": "", "reason": "close_post_opt"})
    executor = RooflineExecutor(shared_state=state)
    with patch(
        "hyperloom.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert captured["payload"].get("roofline_output_name") == "kernel_roofline_opt.json"
    assert captured["payload"].get("roofline_arm") == "current_best"


# --------------------------------------------------------------------------
# Auto-retry returns non-dict (lines 505-506)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_returns_non_dict_fails_and_clears_cache(tmp_path):
    """First trace_analyze fails with a recovery hint; the auto-retry then
    returns a non-dict -> fail with cleared cache (lines 504-513)."""
    fail = {
        "status": "failed",
        "error": "steady_state_chunk_empty",
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_empty",
                "requested_mode": "mixed",
                "non_empty_modes": ["prefilldecode"],
            },
        ],
    }
    calls = {"n": 0}

    async def fake_profile(ctx):
        return _profile_success("/tmp/t.gz")

    async def fake_ta(payload, *, session_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            return fail
        return "definitely-not-a-dict"

    state = _state()
    state.last_trace_analyze = {"analysis_md_text": "stale", "roofline_snapshot_id": 9}
    executor = RooflineExecutor(shared_state=state)
    with patch(
        "hyperloom.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result["error_class"] == "trace_analyze_failed"
    assert "non-dict" in result["error"]
    assert "N26" in result["error"]
    assert "prefilldecode" in result["error"]
    assert calls["n"] == 2
    assert state.last_trace_analyze == {}


# --------------------------------------------------------------------------
# Lifecycle save fast-paths: START (line 262) and END (line 587)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lifecycle_saves_when_session_dir_has_state_json(tmp_path):
    """A real session dir with state.json present triggers both the START
    save (line 262) and the END save (line 587)."""
    session_dir = tmp_path / "sess"
    state = _state()
    _seed_session_dir(session_dir, state)

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 51%\n", encoding="utf-8")

    saves: list[str] = []
    real_save = state.save

    def save_spy(path):
        saves.append(str(path))
        return real_save(path)

    state.save = save_spy  # type: ignore[assignment]

    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), _ta_ok(md))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(session_dir))

    assert result["status"] == "succeeded"
    # START save + END save both hit the real session dir.
    assert saves.count(str(session_dir)) >= 2


# --------------------------------------------------------------------------
# Lifecycle START defensive except (lines 263-264)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lifecycle_start_emit_failure_is_swallowed(tmp_path):
    """record_lifecycle_event raising on the START emit must not abort the
    run (defensive except, lines 263-264)."""
    state = _state()
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    calls = {"n": 0}
    real_evt = state.record_lifecycle_event

    def flaky_evt(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # START emit
            raise RuntimeError("lifecycle START boom")
        return real_evt(*args, **kwargs)

    state.record_lifecycle_event = flaky_evt  # type: ignore[assignment]

    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), _ta_ok(md))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"


# --------------------------------------------------------------------------
# Lifecycle END defensive except (lines 588-589)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lifecycle_end_emit_failure_is_swallowed(tmp_path):
    """record_lifecycle_event raising on the END emit must not fail the run
    (defensive except, lines 588-589)."""
    state = _state()
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    calls = {"n": 0}
    real_evt = state.record_lifecycle_event

    def flaky_evt(*args, **kwargs):
        calls["n"] += 1
        # First call = START (succeeds), second call = END (raises).
        if kwargs.get("status") == "END":
            raise RuntimeError("lifecycle END boom")
        return real_evt(*args, **kwargs)

    state.record_lifecycle_event = flaky_evt  # type: ignore[assignment]

    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), _ta_ok(md))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["n"] >= 2


# --------------------------------------------------------------------------
# _resolve_framework params fast-path (line 634)
# --------------------------------------------------------------------------
def test_resolve_framework_prefers_params(monkeypatch):
    # line 634: params["framework"] wins over env / shared_state.
    monkeypatch.setenv("FRAMEWORK", "vllm")
    state = _state()
    state.framework = "atom"
    exe = RooflineExecutor(shared_state=state)
    ctx = _ctx(params={"framework": "sglang", "base_extra_args": ""})
    assert exe._resolve_framework(ctx) == "sglang"


def test_resolve_framework_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    state = _state()
    state.framework = ""
    exe = RooflineExecutor(shared_state=state)
    ctx = _ctx(params={"base_extra_args": ""})
    assert exe._resolve_framework(ctx) == "vllm"


def test_resolve_framework_falls_back_to_shared_state(monkeypatch):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    state = _state()
    state.framework = "atom"
    exe = RooflineExecutor(shared_state=state)
    ctx = _ctx(params={"base_extra_args": ""})
    assert exe._resolve_framework(ctx) == "atom"


def test_make_roofline_executor_returns_instance():
    state = _state()
    exe = make_roofline_executor(shared_state=state)
    assert isinstance(exe, RooflineExecutor)
    assert exe.shared_state is state
    # module import sanity (silence unused import lint on rf)
    assert rf.RooflineExecutor is RooflineExecutor
