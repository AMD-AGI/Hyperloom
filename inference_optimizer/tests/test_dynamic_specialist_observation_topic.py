# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: dynamic specialist dispatch must not crash the bus.

The free-form dynamic-specialist port (commit 0a983ec3) routed event
names like ``dynamic_specialist_dispatched`` straight into
``MessageBus.append_and_seq`` as the message *topic*. Those names are
not in ``TOPIC_ALLOWLIST``, so the bus raised ``ValueError: unknown
topic`` inside the tick handler, tripping the Coordinator's emergency
stop (16 sessions on 2026-06-08/09). The fix routes them as
``observation`` envelopes carrying the event name under
``payload["kind"]`` (the repo-wide convention).

The tests drive the real Coordinator dispatch handlers against a real
``MessageBus`` (which enforces the topic allowlist), with only the
subprocess dispatch tool stubbed. Before the fix they reproduce the
``unknown topic`` crash; after the fix they pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import coordinator as coord_mod
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.storage import SqliteConnection


@pytest.fixture
def bus(tmp_path) -> MessageBus:
    sc = SqliteConnection(tmp_path / "dyn_spec.db")
    yield MessageBus(sc)
    sc.close()


def _coordinator_with_bus(bus: MessageBus, session_dir: Path) -> Coordinator:
    """Bare Coordinator shell wired only to the bits the dispatch
    handlers touch (``bus`` + ``session_dir``); skip the heavy __init__."""
    coord = Coordinator.__new__(Coordinator)
    coord.bus = bus
    coord.session_dir = session_dir
    return coord


@pytest.fixture
def stub_dispatch_tool(monkeypatch):
    """Stub the subprocess dispatch tool so handlers reach the bus."""
    def _fake(tool_name, tool_input, session_dir):  # noqa: ANN001
        return f"stub::{tool_name}"
    monkeypatch.setattr(
        coord_mod, "execute_dynamic_dispatch_tool", _fake, raising=False,
    )
    # The handlers do a function-local import; patch the source module too.
    import inference_optimizer.orchestrator.dynamic_dispatch_tools as ddt
    monkeypatch.setattr(ddt, "execute_dynamic_dispatch_tool", _fake)
    return _fake


def _intent(**params):
    return Intent(type=IntentType.DELEGATE, payload={"params": params})


@pytest.mark.asyncio
async def test_dispatch_does_not_raise_unknown_topic(bus, tmp_path, stub_dispatch_tool):
    coord = _coordinator_with_bus(bus, tmp_path)
    intent = _intent(tasks=[{"prompt": "do x"}])
    # Before the fix this raises ValueError("unknown topic: ...").
    await coord._handle_dynamic_specialist_dispatch(
        "Orchestration", intent, {"tasks": [{"prompt": "do x"}]},
    )
    msgs = await bus.tail(n=20)
    kinds = {m.payload.get("kind") for m in msgs if m.topic == "observation"}
    assert "dynamic_specialist_dispatched" in kinds


@pytest.mark.asyncio
async def test_dispatch_empty_tasks_emits_error_observation(bus, tmp_path, stub_dispatch_tool):
    coord = _coordinator_with_bus(bus, tmp_path)
    intent = _intent(tasks=[])
    await coord._handle_dynamic_specialist_dispatch("Orchestration", intent, {"tasks": []})
    msgs = await bus.tail(n=20)
    kinds = {m.payload.get("kind") for m in msgs if m.topic == "observation"}
    assert "dynamic_specialist_error" in kinds
