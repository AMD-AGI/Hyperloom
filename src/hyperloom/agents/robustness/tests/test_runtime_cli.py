# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the subprocess transport CLI: the in-process ``_run_tick`` helper and the real ``python -m hyperloom.agents.robustness.runtime.cli tick`` invocation. Pins the request.json/emit.json contract the Coordinator host-side wrapper depends on."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REQUEST_HEARTBEAT = {
    "kind": "coordinator_inbox",
    "session_id": "sess-runtime-1",
    "raw_prompt": (
        "=== Shared session state ===\n"
        "session_id=sess-runtime-1\n"
        "crash_count=0\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    ),
    "context": {"tick_index": 0, "now_unix": 1700000000.0},
}


_REQUEST_HIGH_SEVERITY = {
    "kind": "coordinator_inbox",
    "session_id": "sess-runtime-2",
    "raw_prompt": (
        "=== Shared session state ===\n"
        "session_id=sess-runtime-2\n"
        "crash_count=10\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    ),
}


# ---------------------------------------------------------------------------
# In-process: _run_tick + _coerce_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tick_emits_heartbeat_envelope(tmp_path: Path):
    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {
            **_REQUEST_HEARTBEAT,
            "options": {
                "session_dir": str(tmp_path),
                "auto_probe_inference_server": False,
                # Inert CI hosts have no Ray head; disable the probe so it doesn't mask the heartbeat.
                "ray_probe_enabled": False,
                # CI lacks the TraceLens CLI / WekaFS mounts the external_deps probe expects.
                "external_deps_enabled": False,
            },
        }
    )
    emit = await _run_tick(request)
    assert emit["session_id"] == "sess-runtime-1"
    assert emit["tick_index"] == 1
    envelope = emit["intent_envelope"]
    assert isinstance(envelope, dict) and "intents" in envelope
    intents = envelope["intents"]
    assert len(intents) == 1
    assert intents[0]["intent_type"] == "send_message"
    assert intents[0]["payload"]["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_run_tick_emits_alert_on_high_crash_count(tmp_path: Path):
    """Strategic HIGH symptoms (crash_count=10 → crash_count_emergency) emit alert(high) only."""
    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request({**_REQUEST_HIGH_SEVERITY, "options": {"session_dir": str(tmp_path)}})
    emit = await _run_tick(request)
    intents = emit["intent_envelope"]["intents"]
    intent_types = {i["intent_type"] for i in intents}
    assert "alert" in intent_types
    assert "escalate_strategy_change" not in intent_types


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase_block, expected",
    [
        ("=== Phase ===\nphase     : FRAMEWORK_AGENT\n", "FRAMEWORK_AGENT"),
        # No block is the honest absence of a phase, not a reason to keep the
        # last one: this process is re-entered per tick and a run moves on.
        ("", ""),
    ],
)
async def test_run_tick_publishes_the_phase_the_prompt_carries(tmp_path: Path, phase_block: str, expected: str):
    """RCA spends from a process one below the orchestrator.

    The published phase is a module global, so this interpreter starts with
    none and every RCA call would reach the gateway unplaceable in the run.
    The Coordinator prompt is the only transport that knows the phase here.
    """
    from hyperloom.common.llm_attribution import current_phase, set_current_phase
    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    set_current_phase("STALE_PHASE")
    try:
        request = _coerce_request(
            {
                "kind": "coordinator_inbox",
                "session_id": "sess-phase",
                "raw_prompt": (
                    f"{phase_block}"
                    "=== Shared session state ===\n"
                    "session_id=sess-phase\n"
                    "crash_count=0\n"
                    "=== Inbox for robustness ===\n"
                    "(no new messages)\n"
                ),
                "context": {"tick_index": 0, "now_unix": 1700000000.0},
                "options": {
                    "session_dir": str(tmp_path),
                    "auto_probe_inference_server": False,
                    "ray_probe_enabled": False,
                    "external_deps_enabled": False,
                },
            }
        )
        await _run_tick(request)
        assert current_phase() == expected
    finally:
        set_current_phase("")


@pytest.mark.asyncio
async def test_run_tick_hands_the_parsed_snapshot_to_the_reactor(tmp_path: Path, monkeypatch):
    """Every parsed shared-state field must survive the trip into the reactor."""
    captured: dict[str, object] = {}

    async def _capture_tick(_self, ctx):
        captured["ctx"] = ctx
        return []

    from hyperloom.agents.robustness.role.reactor import Reactor

    monkeypatch.setattr(Reactor, "tick", _capture_tick, raising=True)

    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    prompt = (
        "=== Time budget ===\n"
        "elapsed=116.0min  remaining=4.0min  budget=120min  closing_phase=False\n"
        "=== Shared session state ===\n"
        "tick=42\n"
        "stop_reason=time_exhausted\n"
        "crash_count=3\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    request = _coerce_request(
        {
            "kind": "coordinator_inbox",
            "session_id": "sess-snapshot",
            "raw_prompt": prompt,
            "options": {"session_dir": str(tmp_path)},
        }
    )
    emit = await _run_tick(request)

    assert emit["session_id"] == "sess-snapshot"
    snap = captured["ctx"].shared_state  # type: ignore[union-attr]
    assert snap.tick == 42
    assert snap.stop_reason == "time_exhausted"
    assert snap.crash_count == 3
    assert snap.budget_minutes == 120.0
    assert snap.remaining_minutes == 4.0
    assert snap.elapsed_minutes == 116.0


def test_coerce_request_rejects_bad_kind():
    from hyperloom.agents.robustness.runtime.cli import RuntimeAdapterError, _coerce_request

    with pytest.raises(RuntimeAdapterError):
        _coerce_request({**_REQUEST_HEARTBEAT, "kind": "not-a-real-kind"})


def test_coerce_request_rejects_missing_session_id():
    from hyperloom.agents.robustness.runtime.cli import RuntimeAdapterError, _coerce_request

    bad = dict(_REQUEST_HEARTBEAT)
    bad.pop("session_id")
    with pytest.raises(RuntimeAdapterError):
        _coerce_request(bad)


def test_coerce_request_rejects_empty_raw_prompt():
    from hyperloom.agents.robustness.runtime.cli import RuntimeAdapterError, _coerce_request

    with pytest.raises(RuntimeAdapterError):
        _coerce_request({**_REQUEST_HEARTBEAT, "raw_prompt": "   "})


# ---------------------------------------------------------------------------
# Subprocess: python -m hyperloom.agents.robustness.runtime.cli tick
# ---------------------------------------------------------------------------


def _agent_root() -> Path:
    """Resolve the robustness-agent source root for subprocess PYTHONPATH."""
    return Path(__file__).resolve().parents[1] / "src"


def _run_subprocess(request_obj: dict, request_path: Path, out_path: Path) -> subprocess.CompletedProcess:
    request_path.write_text(
        json.dumps(request_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    env = dict(os.environ)
    src = str(_agent_root())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "hyperloom.agents.robustness.runtime.cli",
            "tick",
            "--request",
            str(request_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_subprocess_tick_emits_heartbeat(tmp_path: Path):
    request_path = tmp_path / "request.json"
    out_path = tmp_path / "emit.json"
    request = {
        **_REQUEST_HEARTBEAT,
        "options": {
            "session_dir": str(tmp_path / "sess"),
            "auto_probe_inference_server": False,
            # CI/dev hosts have no Ray head; disable the probe so it doesn't fire alongside the heartbeat.
            "ray_probe_enabled": False,
            # CI lacks the TraceLens CLI / WekaFS mounts the external_deps probe expects.
            "external_deps_enabled": False,
        },
    }
    proc = _run_subprocess(request, request_path, out_path)
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    emit = json.loads(out_path.read_text(encoding="utf-8"))
    assert emit["session_id"] == "sess-runtime-1"
    intents = emit["intent_envelope"]["intents"]
    assert len(intents) == 1
    assert intents[0]["intent_type"] == "send_message"


def test_subprocess_tick_returns_exit_2_on_invalid_request(tmp_path: Path):
    request_path = tmp_path / "request.json"
    out_path = tmp_path / "emit.json"
    proc = _run_subprocess(
        {"kind": "not-a-real-kind", "session_id": "x", "raw_prompt": "y"},
        request_path,
        out_path,
    )
    assert proc.returncode == 2, f"stdout={proc.stdout} stderr={proc.stderr}"


def test_subprocess_tick_help_smoke():
    """``--help`` should exit 0 — the host validates this before driving."""
    env = dict(os.environ)
    src = str(_agent_root())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src
    proc = subprocess.run(
        [sys.executable, "-m", "hyperloom.agents.robustness.runtime.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    assert "tick" in proc.stdout


# ---------------------------------------------------------------------------
# Multi-node options plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tick_applies_multi_node_options(tmp_path: Path, monkeypatch):
    """``request.options`` overrides land on the per-tick :class:`Config`."""
    # Drop the env so the only non-default Config value comes from options.
    for key in ("ROBUSTNESS_DISABLE_LOCAL_PROBE", "ROBUSTNESS_NODES"):
        monkeypatch.delenv(key, raising=False)

    captured: dict[str, object] = {}

    from hyperloom.agents.robustness.runtime import cli as runtime_cli

    real_build = runtime_cli.build_reactor_components

    def _spy_build(config, *, rca=None, session_id=None):
        captured["config"] = config
        bundle = real_build(config, rca=rca, session_id=session_id)
        return bundle

    async def _zero_tick(_self, _ctx):
        return []

    monkeypatch.setattr(runtime_cli, "build_reactor_components", _spy_build)
    from hyperloom.agents.robustness.role.reactor import Reactor

    monkeypatch.setattr(Reactor, "tick", _zero_tick, raising=True)

    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {
            **_REQUEST_HEARTBEAT,
            "options": {
                "session_dir": str(tmp_path),
                "disable_local_probe": True,
                "nodes": 4,
            },
        }
    )
    await _run_tick(request)

    config = captured["config"]
    assert config.disable_local_probe is True
    assert config.nodes == 4


@pytest.mark.asyncio
async def test_run_tick_surfaces_rca_llm_usage(tmp_path: Path, monkeypatch):
    """A drained RCA usage block is surfaced on the emit payload."""
    from hyperloom.agents.robustness.runtime import cli as runtime_cli

    real_build = runtime_cli.build_reactor_components

    usage = {"input_tokens": 12, "output_tokens": 5, "calls": 1, "latency_ms": 30, "model": "claude-opus-4-7"}

    def _spy_build(config, *, rca=None, session_id=None):
        bundle = real_build(config, rca=rca, session_id=session_id)

        # Replace the rca engine with a stub that reports usage once.
        class _StubRca:
            def drain_usage(self):
                return usage

            async def aclose(self):
                return None

        bundle.components.rca = _StubRca()
        return bundle

    async def _zero_tick(_self, _ctx):
        return []

    monkeypatch.setattr(runtime_cli, "build_reactor_components", _spy_build)
    from hyperloom.agents.robustness.role.reactor import Reactor

    monkeypatch.setattr(Reactor, "tick", _zero_tick, raising=True)

    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {
            **_REQUEST_HEARTBEAT,
            "options": {"session_dir": str(tmp_path)},
        }
    )
    emit = await _run_tick(request)
    assert emit["llm_usage"] == usage


@pytest.mark.asyncio
async def test_run_tick_omits_llm_usage_when_none(tmp_path: Path, monkeypatch):
    """No RCA call → no ``llm_usage`` key on the emit payload."""
    from hyperloom.agents.robustness.runtime import cli as runtime_cli

    real_build = runtime_cli.build_reactor_components

    def _spy_build(config, *, rca=None, session_id=None):
        return real_build(config, rca=rca, session_id=session_id)

    async def _zero_tick(_self, _ctx):
        return []

    monkeypatch.setattr(runtime_cli, "build_reactor_components", _spy_build)
    from hyperloom.agents.robustness.role.reactor import Reactor

    monkeypatch.setattr(Reactor, "tick", _zero_tick, raising=True)

    from hyperloom.agents.robustness.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {
            **_REQUEST_HEARTBEAT,
            "options": {"session_dir": str(tmp_path)},
        }
    )
    emit = await _run_tick(request)
    assert "llm_usage" not in emit
