# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Focused unit tests for roofline executor guard/branch/fallback paths.

Target guard/branch/fallback paths in
``src/hyperloom/orchestrator/actions/executors/roofline.py``. All tests are
hermetic: profile / trace_analyze boundaries are stubbed, filesystem uses
``tmp_path`` only, no GPU / subprocess / network.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hyperloom.orchestrator.actions.executors import roofline as rf
from hyperloom.orchestrator.actions.executors.roofline import (
    RooflineExecutor,
    _extract_steady_state_retry_mode,
    _preflight_probe,
    _profile_err_text,
    _profile_server_log_tail,
    _server_liveness_probe,
    make_roofline_executor,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


# Shared helpers
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
        "hyperloom.orchestrator.actions.executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
        new=fake_ta,
    )


def _seed_session_dir(session_dir: Path, state: SharedState) -> None:
    """Materialize a real state.json so the save() fast-paths execute."""
    session_dir.mkdir(parents=True, exist_ok=True)
    state.save(session_dir)
    assert (session_dir / "state.json").exists()


# _extract_steady_state_retry_mode guards
def test_extract_returns_none_when_warnings_not_a_list():
    res = {"status": "failed", "trace_health_warnings": {"code": "x"}}
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_skips_non_dict_warning_entries():
    # A non-dict entry in the warnings list is skipped; a valid entry after it is still honoured.
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
    # The recovery warning names an alternate field that is not a list -> skip it; a later well-formed warning still
    # matches.
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


# _profile_err_text
def test_profile_err_text_non_dict_returns_empty():
    assert _profile_err_text(None) == ""
    assert _profile_err_text("garbage") == ""
    assert _profile_err_text(42) == ""


def test_profile_err_text_includes_sub_result_fields():
    # A dict sub_result contributes its error / error_class.
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


# _profile_server_log_tail
def test_profile_server_log_tail_non_dict_returns_empty():
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
    # _find_server_logs succeeds but read_bytes raises OSError -> best-effort "" instead of propagating.
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
    # Patch the lazy import target so importing _find_server_logs raises.
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    def boom(_slot):
        raise ImportError("simulated import failure")

    monkeypatch.setattr(br, "_find_server_logs", boom)
    assert _profile_server_log_tail({"workspace": "/nonexistent"}) == ""


def test_profile_server_log_tail_empty_when_no_logs(tmp_path):
    empty = tmp_path / "empty_ws"
    empty.mkdir()
    assert _profile_server_log_tail({"workspace": str(empty)}) == ""


# Preflight and process-level probes
def test_preflight_probe_reports_leftover_traces_and_disk(tmp_path):
    """A trace already sitting in this task's own run directory is reported before anything is profiled."""
    run_dir = tmp_path / "runs" / "roofline" / "t-rf-units"
    run_dir.mkdir(parents=True)
    (run_dir / "old.pt.trace.json.gz").write_bytes(b"leftover")
    (run_dir / "server.log").write_text("not a trace", encoding="utf-8")

    probe = _preflight_probe(tmp_path, "t-rf-units", reaped=[4242])

    assert probe["orphans_reaped"] == [4242]
    assert probe["orphans_reaped_count"] == 1
    assert probe["stale_trace_count"] == 1
    assert probe["stale_traces"][0]["path"] == "runs/roofline/t-rf-units/old.pt.trace.json.gz"
    assert probe["stale_traces"][0]["size_bytes"] == len(b"leftover")
    assert probe["disk"]["free_bytes"] > 0
    assert 0 < probe["disk"]["free_pct"] <= 100


def test_preflight_probe_is_clean_on_a_fresh_session(tmp_path):
    """No runs directory is not a finding, so the leftover list stays empty rather than absent."""
    probe = _preflight_probe(tmp_path, "t-rf-units", reaped=[])

    assert probe["stale_traces"] == []
    assert probe["stale_trace_count"] == 0
    assert probe["orphans_reaped_count"] == 0


def _unused_pid() -> int:
    """A pid no live process holds, so the probe's answer is not a coincidence of the test host."""
    for candidate in range(4_000_000, 4_100_000):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    raise AssertionError("no unused pid available on this host")


def test_server_liveness_probe_flags_a_pidfile_whose_process_is_gone(tmp_path):
    """Teardown unlinks the pidfile, so a pidfile naming a dead pid means the engine died instead."""
    dead = _unused_pid()
    run_dir = tmp_path / "runs" / "roofline" / "t-rf-units"
    run_dir.mkdir(parents=True)
    (run_dir / "sglang_30000.pid").write_text(f"{dead} {dead}\n", encoding="utf-8")
    # Another task's leftover must not be attributed to this one.
    other = tmp_path / "runs" / "roofline" / "t-other"
    other.mkdir(parents=True)
    (other / "sglang_30001.pid").write_text(f"{dead + 1} {dead + 1}\n", encoding="utf-8")

    probe = _server_liveness_probe(tmp_path, "t-rf-units")

    assert probe["pidfiles"] == 1
    assert probe["dead_with_pidfile"] == 1
    assert probe["alive"] == 0
    assert probe["entries"][0]["pid"] == dead
    assert probe["entries"][0]["is_server"] is False


def test_server_liveness_probe_sees_a_live_process_as_not_a_server(tmp_path):
    """A live pid that is not one of our engines is pid reuse, and must not read as a surviving server."""
    run_dir = tmp_path / "runs" / "roofline" / "t-rf-units"
    run_dir.mkdir(parents=True)
    (run_dir / "sglang_30000.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    probe = _server_liveness_probe(tmp_path, "t-rf-units")

    assert probe["alive"] == 1
    assert probe["dead_with_pidfile"] == 0
    assert probe["entries"][0]["is_server"] is False


def test_server_liveness_probe_is_empty_without_a_runs_dir(tmp_path):
    assert _server_liveness_probe(tmp_path, "t-rf-units") == {}


@pytest.mark.asyncio
async def test_executor_records_preflight_and_liveness_on_every_attempt(tmp_path):
    """The probes are wired into the action, not just importable."""
    run_dir = tmp_path / "runs" / "roofline" / "t-rf-units"
    run_dir.mkdir(parents=True)
    (run_dir / "old.pt.trace.json.gz").write_bytes(b"leftover")

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    async def fake_profile(ctx):
        return _profile_success()

    async def fake_ta(payload, *, session_dir):
        return _ta_ok(md)

    recorder = MagicMock()
    with (
        patch("hyperloom.orchestrator.actions.executors.roofline.make_roofline_recorder", return_value=recorder),
        patch("hyperloom.orchestrator.actions.executors.profile.profile_executor", new=fake_profile),
        patch("hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler", new=fake_ta),
    ):
        await RooflineExecutor(shared_state=_state())(_ctx(tmp_path))

    preflight = recorder.record_preflight.call_args.args[0]
    assert preflight["stale_trace_count"] == 1
    assert preflight["disk"]["free_bytes"] > 0
    assert recorder.record_profile_run.call_args.kwargs["server_liveness"] is not None


@pytest.mark.asyncio
async def test_a_non_retryable_profile_failure_still_rows_the_attempt(tmp_path):
    """This branch used to return without recording, so the one failure nobody can retry left no attempt row."""

    async def fake_profile(ctx):
        return {
            "status": "failed",
            "error_class": "primary_rank_trace_missing",
            "error": "no trace for the primary rank",
        }

    async def fake_ta(payload, *, session_dir):
        raise AssertionError("trace_analyze must not run after a non-retryable profile failure")

    recorder = MagicMock()
    with (
        patch("hyperloom.orchestrator.actions.executors.roofline.make_roofline_recorder", return_value=recorder),
        patch("hyperloom.orchestrator.actions.executors.profile.profile_executor", new=fake_profile),
        patch("hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler", new=fake_ta),
    ):
        result = await RooflineExecutor(shared_state=_state())(_ctx(tmp_path))

    assert result["status"] == "failed"
    recorder.record_profile_run.assert_called_once()
    kwargs = recorder.record_profile_run.call_args.kwargs
    assert kwargs["status"] == "failed"
    assert kwargs["failure"]["error_class"] == "primary_rank_trace_missing"
    # And it must not have retried past the attempt it could not recover from.
    assert kwargs["run_index"] == 1


@pytest.mark.asyncio
async def test_instrumentation_is_drained_per_attempt_even_when_absent(tmp_path):
    """The executor is reached through a module-level name a substitute can occupy; the probe must tolerate that."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    async def fake_profile(ctx):
        return _profile_success()

    async def fake_ta(payload, *, session_dir):
        return _ta_ok(md)

    recorder = MagicMock()
    with (
        patch("hyperloom.orchestrator.actions.executors.roofline.make_roofline_recorder", return_value=recorder),
        patch("hyperloom.orchestrator.actions.executors.profile.profile_executor", new=fake_profile),
        patch("hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler", new=fake_ta),
    ):
        await RooflineExecutor(shared_state=_state())(_ctx(tmp_path))

    assert "instrumentation" in recorder.record_profile_run.call_args.kwargs


def test_drain_instrumentation_takes_the_report_once():
    """Draining, not reading: a later attempt must not inherit an earlier attempt's report."""
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
    from hyperloom.orchestrator.actions.executors.roofline import _drain_instrumentation

    executor = ProfileExecutor()
    executor._instrumentation_preflight = {"check_id": "instrumentation_preflight", "status": "passed"}

    assert _drain_instrumentation(executor)["status"] == "passed"
    assert _drain_instrumentation(executor) is None


def test_drain_instrumentation_tolerates_an_executor_without_one():
    from hyperloom.orchestrator.actions.executors.roofline import _drain_instrumentation

    assert _drain_instrumentation(object()) is None

    class _Angry:
        def drain_instrumentation_report(self):
            raise RuntimeError("nope")

    assert _drain_instrumentation(_Angry())["error"].startswith("RuntimeError")


# Exception-path cuda-graph classification
@pytest.mark.asyncio
async def test_profile_exception_with_capture_signature_fails_without_escalating(tmp_path):
    """A capture signature in a raised exception ends the action; it does not arm an eager second attempt."""
    seen: list[dict] = []

    capture_exc = RuntimeError(
        "Capture cuda graph failed: HIP error: operation not permitted "
        "when stream is capturing (hipErrorStreamCaptureUnsupported)"
    )

    async def fake_profile(ctx):
        seen.append(dict(ctx.task.params or {}))
        raise capture_exc

    async def fake_ta(payload, *, session_dir):
        raise AssertionError("trace_analyze must not run after a capture failure")

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_cuda_graph_capture_config_failed"
    assert len(seen) == 1
    assert "--disable-cuda-graph" not in str(seen[0].get("base_extra_args", ""))


# close_post_opt output-name branch
@pytest.mark.asyncio
async def test_close_post_opt_reason_uses_opt_output_name(tmp_path):
    """reason=close_post_opt routes to kernel_roofline_opt.json."""
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
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert captured["payload"].get("roofline_output_name") == "kernel_roofline_opt.json"
    assert captured["payload"].get("roofline_arm") == "current_best"


# Auto-retry returns non-dict
@pytest.mark.asyncio
async def test_retry_returns_non_dict_fails_and_clears_cache(tmp_path):
    """First trace_analyze fails with a recovery hint; the auto-retry then returns a non-dict -> fail with cleared cache."""
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
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result["error_class"] == "trace_analyze_failed"
    assert "non-dict" in result["error"]
    assert "N26" in result["error"]
    assert "prefilldecode" in result["error"]
    assert calls["n"] == 2
    assert state.last_trace_analyze == {}


# Lifecycle save fast-paths: START and END
@pytest.mark.asyncio
async def test_lifecycle_saves_when_session_dir_has_state_json(tmp_path):
    """A real session dir with state.json present triggers both the START save and the END save."""
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
    assert saves.count(str(session_dir)) >= 2


# Lifecycle START defensive except
@pytest.mark.asyncio
async def test_lifecycle_start_emit_failure_is_swallowed(tmp_path, monkeypatch):
    """record_lifecycle_event raising on the START emit must not abort the run."""
    state = _state()
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    calls = {"n": 0}
    real_evt = rf.record_lifecycle_event

    def flaky_evt(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("lifecycle START boom")
        return real_evt(*args, **kwargs)

    monkeypatch.setattr(rf, "record_lifecycle_event", flaky_evt)

    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), _ta_ok(md))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"


# Lifecycle END defensive except
@pytest.mark.asyncio
async def test_lifecycle_end_emit_failure_is_swallowed(tmp_path, monkeypatch):
    """record_lifecycle_event raising on the END emit must not fail the run."""
    state = _state()
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    calls = {"n": 0}
    real_evt = rf.record_lifecycle_event

    def flaky_evt(*args, **kwargs):
        calls["n"] += 1
        if kwargs.get("status") == "END":
            raise RuntimeError("lifecycle END boom")
        return real_evt(*args, **kwargs)

    monkeypatch.setattr(rf, "record_lifecycle_event", flaky_evt)

    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), _ta_ok(md))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["n"] >= 2


# _resolve_framework params fast-path
def test_resolve_framework_prefers_params(monkeypatch):
    # params["framework"] wins over env / shared_state.
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
    assert rf.RooflineExecutor is RooflineExecutor
