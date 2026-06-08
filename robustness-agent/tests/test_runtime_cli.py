# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the subprocess transport CLI: the in-process ``_run_tick`` helper and the real ``python -m robustness_agent.runtime.cli tick`` invocation. Pins the request.json/emit.json contract the Coordinator host-side wrapper depends on."""

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
                # Inert CI hosts have no Ray head: the A6 probe would fire ``ray_head_dead`` and mask the heartbeat envelope.
                "ray_probe_enabled": False,
                # CI lacks the TraceLens CLI / WekaFS mounts the J external_deps probe expects.
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
    """Strategic HIGH symptoms (crash_count_high) emit alert(high) only (escalate/prune auto-emit retired in loosen P3_19)."""
    from robustness_agent.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request(
        {**_REQUEST_HIGH_SEVERITY, "options": {"session_dir": str(tmp_path)}}
    )
    emit = await _run_tick(request)
    intents = emit["intent_envelope"]["intents"]
    intent_types = {i["intent_type"] for i in intents}
    assert "alert" in intent_types
    assert "escalate_strategy_change" not in intent_types


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
            # CI/dev hosts have no Ray head: ``ray status`` hangs and the A6 probe would fire alert/prune alongside the heartbeat.
            "ray_probe_enabled": False,
            # CI lacks the TraceLens CLI / WekaFS mounts the J external_deps probe expects.
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


# ---------------------------------------------------------------------------
# M2 multi-node options plumbing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tick_applies_multi_node_options(tmp_path: Path, monkeypatch):
    """``request.options`` overrides land on the per-tick :class:`Config`."""
    # Drop workload-uid env so the only non-default Config value comes from options.
    for key in (
        "ROBUSTNESS_WORKLOAD_UID",
        "CLAW_WORKLOAD_UID",
        "WORKLOAD_UID",
        "KUBE_WORKLOAD_UID",
        "RAY_JOB_ID",
        "ROBUSTNESS_DISABLE_LOCAL_PROBE",
        "ROBUSTNESS_ENABLE_CLUSTER_POD_METRICS",
        "ROBUSTNESS_NODES",
    ):
        monkeypatch.delenv(key, raising=False)

    captured: dict[str, object] = {}

    from robustness_agent import runtime as runtime_pkg

    real_build = runtime_pkg.cli.build_reactor_components  # type: ignore[attr-defined]

    def _spy_build(config, *, rca=None, session_id=None):
        captured["config"] = config
        bundle = real_build(config, rca=rca, session_id=session_id)
        return bundle

    async def _zero_tick(_self, _ctx):
        return []

    monkeypatch.setattr(runtime_pkg.cli, "build_reactor_components", _spy_build)
    from robustness_agent.role.reactor import Reactor
    monkeypatch.setattr(Reactor, "tick", _zero_tick, raising=True)

    from robustness_agent.runtime.cli import _coerce_request, _run_tick

    request = _coerce_request({
        **_REQUEST_HEARTBEAT,
        "options": {
            "session_dir": str(tmp_path),
            "robustness_server_url": "",
            "disable_local_probe": True,
            "enable_cluster_pod_metrics": True,
            "pod_metrics_categories": "gpu,memory",
            "workload_uid": "wl-123",
            "nodes": 4,
        },
    })
    await _run_tick(request)

    config = captured["config"]
    assert config.disable_local_probe is True
    assert config.enable_cluster_pod_metrics is True
    assert config.pod_metrics_categories == ("gpu", "memory")
    assert config.workload_uid == "wl-123"
    assert config.nodes == 4
