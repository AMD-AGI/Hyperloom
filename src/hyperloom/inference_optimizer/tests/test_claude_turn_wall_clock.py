# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Claude SDK call that never finishes is cut off at its wall-clock bound, and its CLI with it.

The idle timeout between streamed messages cannot bound a turn: the CLI reports each retry of its own API request
as a message, so a stalled gateway keeps the stream "active" for as long as the CLI keeps retrying.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import DEFAULT_REACTOR_TURN_TIMEOUT_SEC, Coordinator
from hyperloom.orchestrator.roles import ClaudeBackend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.roles import claude as claude_mod
from hyperloom.orchestrator.roles.base import LLMCallFailed, RetryPolicy


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _ApiRetry:
    """Shape of the SDK's ``SystemMessage`` for the CLI's ``api_retry`` frame."""

    subtype = "api_retry"

    def __init__(self, attempt: int) -> None:
        self.data = {"attempt": attempt, "max_retries": 10, "error_status": None, "error": "timeout"}


class _StallingSdk:
    """Fake ``query()``: ``mode`` picks how the stream stalls; records what happened to each call."""

    def __init__(self, mode: str, *, child_cmd: list[str] | None = None) -> None:
        self.mode = mode
        self.child_cmd = child_cmd
        self.started = 0
        self.closed = 0
        self.children: list[subprocess.Popen[bytes]] = []
        self.release_close = asyncio.Event()

    def __call__(self, *, prompt: str, options: _Options):
        self.started += 1
        return self._stream(options)

    async def _stream(self, options: _Options):
        if self.child_cmd is not None:
            # A CLI the SDK failed to reap: its own process group, the turn's environment, and never killed here.
            env = {**os.environ, **options.kwargs.get("env", {})}
            self.children.append(subprocess.Popen(self.child_cmd, env=env, start_new_session=True))
        try:
            if self.mode in ("silent", "wedged_close_silent"):
                await asyncio.sleep(3600)
            if self.mode == "dies_on_cancel":
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    # The cancellation never surfaces: the stream fails as one whose CLI was killed under it would.
                    raise OSError("CLI exited") from None
            n = 0
            busy_until = time.monotonic() + 5.0
            # "retrying": a message well inside the idle budget, well past every bound a test sets; then the stream
            # ends, so one that a regression leaves running cannot outlive the test's event loop as well.
            frames_until = time.monotonic() + 25.0
            while time.monotonic() < frames_until:
                n += 1
                yield _ApiRetry(n)
                # "busy": back to back for a few seconds, so a cancellation lands as a message does. Before Python
                # 3.12 ``asyncio.wait_for`` swallows such a cancellation, and the turn runs on.
                busy = self.mode == "busy" and time.monotonic() < busy_until
                await asyncio.sleep(0 if busy else 0.01)
        finally:
            if self.mode in ("wedged_close", "wedged_close_silent"):
                # A close that ignores cancellation, as a transport.close() that never returns would.
                while not self.release_close.is_set():
                    try:
                        await self.release_close.wait()
                    except asyncio.CancelledError:
                        continue
            self.closed += 1


async def _guarded(turn: Any) -> Any:
    """Keep a turn that is not bounded from hanging the suite; the assertions say which bound ended it.

    Waits without cancelling: cancelling a turn whose close ignores cancellation would hang here instead.
    """
    task = asyncio.ensure_future(turn)
    done, _ = await asyncio.wait({task}, timeout=20.0)
    if not done:
        pytest.fail("the turn was still running 20 s after its bound")
    return task.result()


def _backend(sdk: _StallingSdk, *, turn_timeout_s: float, call_timeout_s: float = 30.0) -> ClaudeBackend:
    return ClaudeBackend(
        sdk_query_factory=sdk,
        sdk_options_cls=_Options,
        enable_mcp_emit_intent=False,
        call_timeout_s=call_timeout_s,
        turn_timeout_s=turn_timeout_s,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_s=0.0),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retrying", "busy", "silent"])
async def test_a_stalled_turn_ends_at_its_wall_clock_bound_and_closes_the_stream(mode: str, caplog, monkeypatch):
    # A grace well inside the guard: a stop that is not taken shows as an abandoned close, not as a guard timeout.
    monkeypatch.setattr(claude_mod, "_TURN_CLEANUP_GRACE_SEC", 5.0)
    sdk = _StallingSdk(mode)
    backend = _backend(sdk, turn_timeout_s=0.3)

    started = time.monotonic()
    with caplog.at_level("WARNING"), pytest.raises(LLMCallFailed) as raised:
        await _guarded(backend.run("hi"))
    elapsed = time.monotonic() - started

    # Ended by the 0.3 s wall-clock bound, not by the 30 s idle budget nor by a retry.
    assert 0.3 <= elapsed < 5.0
    assert sdk.started == 1
    assert sdk.closed == 1
    assert "exceeded its 0.3s wall-clock bound (INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC)" in str(raised.value)
    assert "SDK closed the CLI" in str(raised.value)
    assert "claude SDK turn exceeded its 0.3s wall-clock bound" in caplog.text


@pytest.mark.asyncio
async def test_the_cli_retry_frames_are_logged_while_they_hold_the_stream_open(caplog):
    sdk = _StallingSdk("retrying")
    backend = _backend(sdk, turn_timeout_s=0.2)

    with caplog.at_level("WARNING"), pytest.raises(LLMCallFailed):
        await _guarded(backend.run("hi"))

    assert "claude CLI retrying its API request (attempt 1/10, status=None, error=timeout)" in caplog.text


@pytest.mark.asyncio
async def test_a_close_that_never_finishes_is_abandoned_after_the_grace(monkeypatch):
    monkeypatch.setattr(claude_mod, "_TURN_CLEANUP_GRACE_SEC", 0.2)
    sdk = _StallingSdk("wedged_close")
    backend = _backend(sdk, turn_timeout_s=0.2)

    started = time.monotonic()
    try:
        with pytest.raises(LLMCallFailed) as raised:
            await _guarded(backend.run("hi"))
        elapsed = time.monotonic() - started

        assert elapsed < 5.0
        assert "SDK close still running after 0.2s, abandoned" in str(raised.value)
        # Held, not dropped, while its close is still running; released once it finishes.
        assert len(backend._abandoned_turns) == 1
    finally:
        sdk.release_close.set()
    for _ in range(100):
        if not backend._abandoned_turns:
            break
        await asyncio.sleep(0.01)
    assert backend._abandoned_turns == set()
    assert sdk.closed == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="finds the turn's processes through /proc")
async def test_the_cli_left_behind_by_a_timed_out_turn_is_killed_and_nothing_else_is():
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    sdk = _StallingSdk("silent", child_cmd=sleeper)
    backend = _backend(sdk, turn_timeout_s=0.5)
    bystander = subprocess.Popen(sleeper, start_new_session=True)
    try:
        with pytest.raises(LLMCallFailed) as raised:
            await _guarded(backend.run("hi"))

        (cli,) = sdk.children
        assert await asyncio.to_thread(cli.wait, 10) == -9
        assert "killed 1 leftover CLI process(es)" in str(raised.value)
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()
        for child in sdk.children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="finds the turn's processes through /proc")
async def test_the_clis_left_behind_by_a_turn_that_fails_on_its_own_are_killed():
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    sdk = _StallingSdk("silent", child_cmd=sleeper)
    # Every attempt ends on its idle timeout, well inside the wall-clock bound.
    backend = _backend(sdk, turn_timeout_s=600.0, call_timeout_s=0.1)
    bystander = subprocess.Popen(sleeper, start_new_session=True)
    try:
        with pytest.raises(LLMCallFailed, match="stream idle"):
            await _guarded(backend.run("hi"))

        assert sdk.started == 3
        assert [await asyncio.to_thread(cli.wait, 10) for cli in sdk.children] == [-9, -9, -9]
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()
        for child in sdk.children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="finds the turn's processes through /proc")
async def test_a_turn_cancelled_by_its_caller_still_kills_its_cli():
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    sdk = _StallingSdk("silent", child_cmd=sleeper)
    backend = _backend(sdk, turn_timeout_s=600.0)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(backend.run("hi"), timeout=0.5)  # the caller's own bound
        # The SDK call closes on the loop after its caller has gone; let it.
        for _ in range(500):
            if sdk.closed:
                break
            await asyncio.sleep(0.01)

        (cli,) = sdk.children
        # Waited off the loop: the kill runs on it, after the SDK call has closed.
        assert await asyncio.to_thread(cli.wait, 10) == -9
        assert sdk.closed == 1
    finally:
        for child in sdk.children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.asyncio
async def test_a_turn_that_does_not_take_its_cancellation_starts_no_further_cli(monkeypatch):
    monkeypatch.setattr(claude_mod, "_TURN_CLEANUP_GRACE_SEC", 5.0)
    sdk = _StallingSdk("dies_on_cancel")
    backend = _backend(sdk, turn_timeout_s=0.3)

    with pytest.raises(LLMCallFailed) as raised:
        await _guarded(backend.run("hi"))

    # The failed stream is retryable, but a turn being stopped never starts another CLI.
    assert sdk.started == 1
    assert "SDK closed the CLI" in str(raised.value)
    assert backend._abandoned_turns == set()


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="finds the turn's processes through /proc")
async def test_a_cancelled_caller_whose_close_never_finishes_still_gets_its_cli_killed(monkeypatch):
    monkeypatch.setattr(claude_mod, "_TURN_CLEANUP_GRACE_SEC", 0.5)
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    sdk = _StallingSdk("wedged_close_silent", child_cmd=sleeper)
    backend = _backend(sdk, turn_timeout_s=600.0)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(backend.run("hi"), timeout=0.5)

        (cli,) = sdk.children
        # Killed after the grace, although the SDK call is still closing.
        assert await asyncio.to_thread(cli.wait, 10) == -9
        assert sdk.closed == 0
        assert len(backend._abandoned_turns) == 1
    finally:
        sdk.release_close.set()
        for child in sdk.children:
            if child.poll() is None:
                child.kill()
                child.wait()
    for _ in range(100):
        if not backend._abandoned_turns:
            break
        await asyncio.sleep(0.01)
    assert backend._abandoned_turns == set()
    assert backend._turn_stoppers == set()


def test_each_turn_tags_its_cli_with_its_own_marker():
    backend = _backend(_StallingSdk("silent"), turn_timeout_s=1.0)
    env_a = backend._build_options(tools=[], max_turns=8, system_prompt=None, turn_marker="a").kwargs["env"]
    env_b = backend._build_options(tools=[], max_turns=8, system_prompt=None, turn_marker="b").kwargs["env"]
    assert env_a["HYPERLOOM_CLAUDE_TURN_ID"] == "a"
    assert env_b["HYPERLOOM_CLAUDE_TURN_ID"] == "b"


def test_turn_bound_default_and_env_override(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC", raising=False)
    default = ClaudeBackend(sdk_query_factory=_StallingSdk("silent"), sdk_options_cls=_Options).turn_timeout_s
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC", "42.5")
    configured = ClaudeBackend(sdk_query_factory=_StallingSdk("silent"), sdk_options_cls=_Options).turn_timeout_s
    assert (default, configured) == (1500.0, 42.5)
    # The backend gives up, tears the CLI down and reports first, before the reactor stage bound cancels it.
    assert default + claude_mod._TURN_CLEANUP_GRACE_SEC < DEFAULT_REACTOR_TURN_TIMEOUT_SEC


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retrying", "busy"])
async def test_the_tick_loop_keeps_ticking_past_a_hung_orchestration_turn(mode: str, session_dir, monkeypatch):
    monkeypatch.setattr(claude_mod, "_TURN_CLEANUP_GRACE_SEC", 5.0)
    monkeypatch.setattr("hyperloom.orchestrator.loop.coordinator._BACKEND_RETRY_BASE_SEC", 0)
    sdk = _StallingSdk(mode)
    orchestration = _backend(sdk, turn_timeout_s=0.3)
    critic = MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="critic")
    coord = Coordinator(session_dir, backends={"orchestration": orchestration, "critic": critic})
    coord.shared_state.baseline_tput = 800.0
    # Only the backend's own bound may end the turn here.
    coord.reactor_turn_timeout_sec = 600.0
    observations: list[dict[str, Any]] = []
    record = coord.writeback.record_observation

    async def _capture(sender: str, kind: str, payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        observations.append(payload)
        return await record(sender, kind, payload, *args, **kwargs)

    coord.writeback.record_observation = _capture  # type: ignore[method-assign]
    try:
        reason = await asyncio.wait_for(coord.run(max_ticks=2, tick_interval_sec=0.0), timeout=30.0)
    finally:
        await coord.stop()

    assert reason == "max_ticks"
    assert (sdk.started, sdk.closed) == (2, 2)
    errors = [o for o in observations if o.get("kind") == "backend_error" and o.get("agent") == "orchestration"]
    assert len(errors) == 2
    assert all("wall-clock bound" in o["error"] and "SDK closed the CLI" in o["error"] for o in errors)
