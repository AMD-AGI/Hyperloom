# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD engine profiling by step count per role, or by a timed window."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as life
from hyperloom.orchestrator.actions.executors import profile

_PD_WORKERS = {
    "backend": "sglangrouter",
    "prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0", "enginePort": 8010}],
    "decode_pods": [{"podIP": "10.0.0.2", "podId": "d0", "enginePort": 8020}],
}
_ROLE_BY_PORT = {8010: "prefill", 8020: "decode"}


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch):
    """Route PD engine requests to a mock transport; return the ``(role, path, body)`` log and the status per path."""
    seen: list[tuple[str, str, dict]] = []
    status: dict[str, int] = {}
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((_ROLE_BY_PORT[request.url.port], request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(status.get(f"{_ROLE_BY_PORT[request.url.port]}{request.url.path}", 200))

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    for key in (
        "HYPERLOOM_MN_PD_PROFILE_MODE",
        "HYPERLOOM_MN_PD_PREFILL_PROFILE_STEPS",
        "HYPERLOOM_MN_PD_DECODE_PROFILE_STEPS",
        "HYPERLOOM_MN_PROFILE_WITH_STACK",
    ):
        monkeypatch.delenv(key, raising=False)
    return seen, status


def _starts(seen: list[tuple[str, str, dict]]) -> dict[str, dict]:
    return {role: body for role, path, body in seen if path == "/start_profile"}


def test_steps_mode_is_the_default_and_sends_each_role_its_step_window(engines, tmp_path: Path) -> None:
    seen, _status = engines

    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))

    assert _starts(seen) == {
        "prefill": {
            "num_steps": 8,
            "profile_by_stage": True,
            "record_shapes": True,
            "with_stack": False,
            "output_dir": str(tmp_path / "prefill" / "0"),
        },
        "decode": {
            "num_steps": 256,
            "profile_by_stage": False,
            "record_shapes": True,
            "with_stack": False,
            "output_dir": str(tmp_path / "decode" / "0"),
        },
    }


def test_steps_mode_step_counts_and_stack_follow_env(engines, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen, _status = engines
    monkeypatch.setenv("HYPERLOOM_MN_PD_PREFILL_PROFILE_STEPS", "4")
    monkeypatch.setenv("HYPERLOOM_MN_PD_DECODE_PROFILE_STEPS", "16")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WITH_STACK", "1")

    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))

    starts = _starts(seen)
    assert (starts["prefill"]["num_steps"], starts["decode"]["num_steps"]) == (4, 16)
    assert starts["prefill"]["with_stack"] is True and starts["decode"]["with_stack"] is True


def test_time_mode_sends_only_the_output_dir_and_stack_flag(
    engines, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen, _status = engines
    monkeypatch.setenv("HYPERLOOM_MN_PD_PROFILE_MODE", "time")

    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))
    asyncio.run(life.trigger_pd_engine_profile("stop"))

    assert seen == [
        ("prefill", "/start_profile", {"output_dir": str(tmp_path / "prefill" / "0"), "with_stack": False}),
        ("decode", "/start_profile", {"output_dir": str(tmp_path / "decode" / "0"), "with_stack": False}),
        ("prefill", "/stop_profile", {}),
        ("decode", "/stop_profile", {}),
    ]


def test_start_returns_the_roles_that_accepted_and_an_error_stop_is_not_raised(engines, tmp_path: Path) -> None:
    _seen, status = engines
    status["prefill/start_profile"] = 500
    status["decode/stop_profile"] = 500

    started = asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))
    asyncio.run(life.trigger_pd_engine_profile("stop"))

    assert started == ["decode"]


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch):
    """Record the Infera/PD profile calls and trace polls the profile window makes; return the event log."""
    events: list[str] = []
    polls: list[list[dict]] = [[]]

    async def infera(action: str, body: dict | None = None) -> None:
        events.append(f"infera:{action}")

    async def pd(action: str, body: dict | None = None) -> list[str]:
        events.append(f"pd:{action}")
        return ["prefill", "decode"] if action == "start" else []

    def select(root: str, since_unix: float) -> dict:
        events.append("poll")
        script = polls[0]
        return script.pop(0) if len(script) > 1 else (script[0] if script else {})

    monkeypatch.setattr(life, "trigger_infera_engine_profile", infera)
    monkeypatch.setattr(life, "trigger_pd_engine_profile", pd)
    monkeypatch.setattr(life, "select_pd_node_traces", select)
    monkeypatch.setattr(profile, "_PD_TRACE_POLL_S", 0.01)
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WARMUP_S", "0")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WINDOW_S", "0")
    monkeypatch.delenv("HYPERLOOM_MN_PD_PROFILE_MODE", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_STOP_PROFILE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)
    return events, polls


async def _run_body(root: str, *, body_s: float = 0.05, fail: bool = False) -> None:
    async with profile._infera_profile_window(root, 0.0):
        await asyncio.sleep(body_s)
        if fail:
            raise RuntimeError("benchmark failed")


_BOTH_ROLES = {"prefill": ["/t/prefill/a.trace.json.gz"], "decode": ["/t/decode/b.trace.json.gz"]}


def test_time_mode_window_stops_every_engine_after_the_window(window, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _polls = window
    monkeypatch.setenv("HYPERLOOM_MN_PD_PROFILE_MODE", "time")

    asyncio.run(_run_body("/t"))

    assert events == ["infera:start", "pd:start", "infera:stop", "pd:stop"]


def test_steps_mode_window_has_no_timed_pd_stop_and_stops_once_traces_appear(window) -> None:
    events, polls = window
    polls[0] = [{}, {"decode": ["/t/decode/b.trace.json.gz"]}, _BOTH_ROLES]

    asyncio.run(_run_body("/t"))

    assert events == ["infera:start", "pd:start", "infera:stop", "poll", "poll", "poll", "pd:stop"]


def test_steps_mode_safety_stop_is_sent_when_traces_never_appear(window, monkeypatch: pytest.MonkeyPatch) -> None:
    events, polls = window
    polls[0] = [{}]
    monkeypatch.setenv("HYPERLOOM_MN_STOP_PROFILE_TIMEOUT_S", "0.05")

    asyncio.run(_run_body("/t"))

    assert events[:3] == ["infera:start", "pd:start", "infera:stop"]
    assert events.count("poll") >= 2
    assert events[-1] == "pd:stop" and events.count("pd:stop") == 1


def test_steps_mode_failed_body_sends_the_safety_stop_without_waiting(window) -> None:
    events, _polls = window

    with pytest.raises(RuntimeError, match="benchmark failed"):
        asyncio.run(_run_body("/t", fail=True))

    assert "poll" not in events
    assert events[-1] == "pd:stop" and events.count("pd:stop") == 1


def test_steps_mode_body_done_before_warmup_sends_no_pd_request(window, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _polls = window
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WARMUP_S", "60")

    asyncio.run(_run_body("/t", body_s=0.0))

    assert events == []
