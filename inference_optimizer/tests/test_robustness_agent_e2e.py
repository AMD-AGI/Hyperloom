# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end test: real robustness-agent runtime + Coordinator.

Mirrors :mod:`test_p2_critic_agent_e2e` for the robustness role: actually
shells out to ``python -m robustness_agent.runtime.cli tick`` against the
real ``robustness-agent`` checkout (sibling of ``inference_optimizer/``)
and verifies the Coordinator surfaces emitted intents on the bus and on
disk.

Marker: ``robustness_agent_e2e`` — devs without a robustness-agent
checkout in their tree skip via ``pytest -m 'not robustness_agent_e2e'``.

Verifies:

* Heartbeat path (zero crash, no inbox): real runtime emits a
  ``send_message{topic=heartbeat}`` envelope; Coordinator records it on
  the bus.
* Alert path (high crash count): real runtime emits
  ``alert(severity=high)`` + ``escalate_strategy_change``; both land on
  the bus with ``from=robustness``.
* Filesystem audit: per-turn
  ``<session_dir>/robustness-workdir/000000/{request.json,emit.json}``
  exists; ``emit.json`` carries a valid ``intent_envelope`` whose schema
  matches what :func:`validate_envelope` would accept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.cli import _resolve_robustness_agent_root
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    RobustnessAgentBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


pytestmark = pytest.mark.robustness_agent_e2e


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture
def robustness_agent_root() -> Path:
    """Locate the real robustness-agent checkout. Skip gracefully if absent."""
    root = _resolve_robustness_agent_root()
    if root is None:
        pytest.skip(
            "robustness-agent runtime not found — set ROBUSTNESS_AGENT_ROOT or "
            "place robustness-agent/ next to inference_optimizer/"
        )
    return root


def _heartbeat_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _seed_state(session_dir: Path, *, crash_count: int) -> SharedState:
    state = SharedState(
        session_id=session_dir.name,
        model_name="test-model",
        model_class="test",
        crash_count=crash_count,
    )
    state.save(session_dir)
    return state


@pytest.mark.asyncio
async def test_robustness_agent_real_runtime_heartbeat(
    session_dir: Path, robustness_agent_root: Path,
):
    """Zero crash + empty inbox → real runtime emits a heartbeat envelope."""
    _seed_state(session_dir, crash_count=0)

    backend = RobustnessAgentBackend(
        robustness_agent_root=robustness_agent_root,
        session_dir=session_dir,
        # IMPORTANT: do NOT pass runtime_caller_factory — we want the
        # real subprocess path here.
        # Heartbeat path: explicitly disable the LocalProbe family
        # probes so an inert CI host (no inference server / no Ray
        # head on 127.0.0.1) doesn't fire ``local_server_unreachable``
        # / ``ray_head_dead`` HIGH alerts that would mask the expected
        # heartbeat ``send_message``. ``ray_probe_enabled`` was added
        # by PR #239 and defaults to True; the e2e heartbeat test
        # needs both off.
        options={
            "robustness_server_url": "",
            "auto_probe_inference_server": False,
            "ray_probe_enabled": False,
            # CI runners lack the TraceLens CLI and WekaFS mounts the
            # J external_deps probe expects; disable it so the heartbeat
            # is not buried under ``tracelens_cli_missing`` /
            # ``wekafs_degraded`` alerts.
            "external_deps_enabled": False,
        },
    )

    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[], default_intent=_heartbeat_intent()),
            name="orchestration",
        ),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": backend,
    }
    c = Coordinator(session_dir, backends=backends)

    try:
        await c.tick(2)
    finally:
        await c.stop()

    assert backend.calls, "robustness backend was not driven"
    workdir = session_dir / "robustness-workdir"
    assert workdir.is_dir()
    turn0 = workdir / "000000"
    for fname in ("request.json", "emit.json"):
        assert (turn0 / fname).is_file(), f"missing per-turn artefact {fname}"

    request = json.loads((turn0 / "request.json").read_text())
    assert request["kind"] == "coordinator_inbox"
    assert request["session_id"] == session_dir.name
    assert "raw_prompt" in request and request["raw_prompt"].strip()
    assert request["options"]["session_dir"] == str(session_dir)

    emit = json.loads((turn0 / "emit.json").read_text())
    envelope = emit["intent_envelope"]
    assert "intents" in envelope and isinstance(envelope["intents"], list)
    assert any(
        i["intent_type"] == "send_message"
        and i["payload"].get("topic") == "heartbeat"
        for i in envelope["intents"]
    ), f"heartbeat missing from emit envelope: {envelope}"


@pytest.mark.asyncio
async def test_robustness_agent_real_runtime_emits_alert_on_high_crash(
    session_dir: Path, robustness_agent_root: Path,
):
    """``crash_count >= 10`` flips the reactor into the high-severity ladder
    rung; the runtime must emit ``alert(high)`` + ``escalate_strategy_change``
    and the Coordinator must mirror both onto the bus with ``from=robustness``."""
    _seed_state(session_dir, crash_count=10)

    backend = RobustnessAgentBackend(
        robustness_agent_root=robustness_agent_root,
        session_dir=session_dir,
        options={"robustness_server_url": ""},
    )

    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[], default_intent=_heartbeat_intent()),
            name="orchestration",
        ),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": backend,
    }
    c = Coordinator(session_dir, backends=backends)

    try:
        await c.tick(2)
        alerts = await c.bus.tail(topic="alert")
        assert alerts, "expected at least one alert on the bus"
        from_robustness = [a for a in alerts if a.from_agent == "robustness"]
        assert from_robustness, (
            f"expected an alert with from=robustness, got "
            f"{[(a.from_agent, a.payload) for a in alerts]}"
        )
        severities = {a.payload.get("severity") for a in from_robustness}
        assert "high" in severities, (
            f"expected severity=high in {severities!r} for alerts "
            f"{[a.payload for a in from_robustness]}"
        )
    finally:
        await c.stop()

    workdir = session_dir / "robustness-workdir" / "000000"
    emit = json.loads((workdir / "emit.json").read_text())
    intent_types = {i["intent_type"] for i in emit["intent_envelope"]["intents"]}
    assert "alert" in intent_types
    assert "escalate_strategy_change" in intent_types, (
        f"high-severity ladder must escalate, got {intent_types}"
    )


@pytest.mark.asyncio
async def test_robustness_agent_workdir_is_per_turn(
    session_dir: Path, robustness_agent_root: Path,
):
    """Each Coordinator tick allocates a new ``<turn>/`` subdir."""
    _seed_state(session_dir, crash_count=0)

    backend = RobustnessAgentBackend(
        robustness_agent_root=robustness_agent_root,
        session_dir=session_dir,
        options={"robustness_server_url": ""},
    )

    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[], default_intent=_heartbeat_intent()),
            name="orchestration",
        ),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": backend,
    }
    c = Coordinator(session_dir, backends=backends)

    try:
        await c.tick(3)
    finally:
        await c.stop()

    workdir = session_dir / "robustness-workdir"
    turns = sorted(p.name for p in workdir.iterdir() if p.is_dir())
    assert turns == ["000000", "000001", "000002"], (
        f"per-turn workdir layout broken: {turns}"
    )
    for t in turns:
        for fname in ("request.json", "emit.json"):
            assert (workdir / t / fname).is_file(), (
                f"missing {fname} under {t}"
            )
