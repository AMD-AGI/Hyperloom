"""Tests for the subprocess transport CLI.

Exercises both the in-process ``_run_tick`` helper (fast) and the real
``python -m robustness_agent.runtime.cli tick`` invocation (slow but
catches packaging / argparse regressions).

The host-side wrapper in ``inference_optimizer/orchestrator/backends/
robustness_agent.py`` builds the same ``request.json`` and reads the
same ``emit.json`` shape this file exercises, so the contract checked
here is the contract the Coordinator depends on.
"""

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
    from robustness_agent.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {
            **_REQUEST_HEARTBEAT,
            "options": {
                "session_dir": str(tmp_path),
                "auto_probe_inference_server": False,
                # Heartbeat path runs on inert CI hosts without a Ray
                # head. Without this opt-out the LocalProbe A6 sub-probe
                # fires ``ray_head_dead`` after the 5s timeout and the
                # ladder appends alert + prune_branch +
                # escalate_strategy_change intents that mask the
                # expected ``send_message{heartbeat}`` envelope.
                "ray_probe_enabled": False,
                # CI containers lack the TraceLens CLI and WekaFS mounts
                # the J external_deps probe expects. Disable the whole
                # probe so it does not enqueue ``tracelens_cli_missing``
                # / ``wekafs_degraded`` alerts alongside the heartbeat.
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
    from robustness_agent.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {**_REQUEST_HIGH_SEVERITY, "options": {"session_dir": str(tmp_path)}}
    )
    emit = await _run_tick(request)
    intents = emit["intent_envelope"]["intents"]
    intent_types = {i["intent_type"] for i in intents}
    assert "alert" in intent_types
    assert "escalate_strategy_change" in intent_types


@pytest.mark.asyncio
async def test_run_tick_propagates_session_id_when_prompt_lacks_it(tmp_path: Path):
    from robustness_agent.runtime.cli import _coerce_request, _run_tick

    prompt = (
        "=== Shared session state ===\n"
        "crash_count=0\n"
        "=== Inbox for robustness ===\n"
        "(no new messages)\n"
    )
    request = _coerce_request({
        "kind": "coordinator_inbox",
        "session_id": "sess-fallback",
        "raw_prompt": prompt,
        "options": {"session_dir": str(tmp_path)},
    })
    emit = await _run_tick(request)
    assert emit["session_id"] == "sess-fallback"


def test_coerce_request_rejects_bad_kind():
    from robustness_agent.runtime.cli import RuntimeAdapterError, _coerce_request

    with pytest.raises(RuntimeAdapterError):
        _coerce_request({**_REQUEST_HEARTBEAT, "kind": "not-a-real-kind"})


def test_coerce_request_rejects_missing_session_id():
    from robustness_agent.runtime.cli import RuntimeAdapterError, _coerce_request

    bad = dict(_REQUEST_HEARTBEAT)
    bad.pop("session_id")
    with pytest.raises(RuntimeAdapterError):
        _coerce_request(bad)


def test_coerce_request_rejects_empty_raw_prompt():
    from robustness_agent.runtime.cli import RuntimeAdapterError, _coerce_request

    with pytest.raises(RuntimeAdapterError):
        _coerce_request({**_REQUEST_HEARTBEAT, "raw_prompt": "   "})


# ---------------------------------------------------------------------------
# Subprocess: python -m robustness_agent.runtime.cli tick
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
        [sys.executable, "-m", "robustness_agent.runtime.cli", "tick",
         "--request", str(request_path), "--out", str(out_path)],
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
            # Heartbeat path is exercised on CI/dev hosts with no Ray
            # head running, where ``ray status`` hangs / times out and
            # the LocalProbe ladder would otherwise fire alert + prune
            # intents alongside the heartbeat. Disable the A6 probe so
            # the envelope stays focused on the heartbeat contract.
            "ray_probe_enabled": False,
            # Same rationale for the J external_deps probe — CI lacks
            # the TraceLens CLI / WekaFS mounts it expects.
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
        [sys.executable, "-m", "robustness_agent.runtime.cli", "--help"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    assert "tick" in proc.stdout
