# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The emit helpers every emit point calls, and the data schema the emitters must stay consistent with."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from hyperloom.plugs import Implementation, ManifestSources, PlugHost, activated, builtin_registry, resolve_manifest
from hyperloom.plugs import progress
from hyperloom.plugs.host import EVENT_LOCK_MISMATCH
from hyperloom.plugs.manifest import LockMismatch, ResolvedManifest, SocketSetting

_SCHEMA = json.loads(
    (Path(progress.__file__).parents[1] / "contracts" / "hyperloom_events.v1.schema.json").read_text(encoding="utf-8")
)

# The v1 type set shared with the platform: nothing is added without a schema entry, nothing dropped without one.
_V1_TYPES = {
    "hyperloom.campaign_started",
    "hyperloom.plugs_lock_mismatch",
    "hyperloom.phase_changed",
    "hyperloom.measurement_started",
    "hyperloom.measurement_finished",
    "hyperloom.node_quarantined",
    "hyperloom.candidate_settled",
    "hyperloom.best_updated",
    "hyperloom.budget",
    "hyperloom.warning",
    "hyperloom.campaign_finished",
    "hyperloom.campaign_failed",
}


class _Spy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, data) -> None:
        self.events.append((event_type, dict(data)))

    def flush(self) -> None:
        pass


class _Raising(_Spy):
    def emit(self, event_type, data) -> None:
        raise RuntimeError("sink is broken")


def _host(impl) -> PlugHost:
    registry = builtin_registry()
    registry.register(Implementation("events", "spy", lambda ctx: impl))
    manifest = resolve_manifest(registry, ManifestSources(cli_plugs=("events=spy",)))
    host = PlugHost(manifest, registry, session_dir=None)
    host.start()
    return host


@pytest.fixture(autouse=True)
def _fresh_budget_clock(monkeypatch):
    monkeypatch.setattr(progress._budget, "last", None)


def _state(minutes: float) -> SimpleNamespace:
    return SimpleNamespace(elapsed_minutes=lambda: minutes, max_minutes=60)


def _valid(event_type: str, data: dict) -> None:
    jsonschema.validate(data, _SCHEMA["events"][event_type])


def test_the_schema_covers_exactly_the_v1_types():
    assert set(_SCHEMA["events"]) == _V1_TYPES
    jsonschema.Draft202012Validator.check_schema({"type": "object", "properties": _SCHEMA["events"]})


def test_what_the_emit_helpers_send_matches_the_schema():
    spy = _Spy()
    with activated(_host(spy)):
        progress.phase_changed("", "BASELINE", 0)
        progress.phase_changed("BASELINE", "FRAMEWORK_AGENT", 2)
        progress.candidate_settled("t-1", "explore", "keep")
        progress.candidate_settled("t-2", "", "failed", 3.5)
        progress.best_updated(1100.0, SimpleNamespace(baseline_tput=1000.0, optimization_stack=[1, 2]), "req-1")
        progress.best_updated(1100.0, SimpleNamespace(baseline_tput=0.0, optimization_stack=[]), None)
        progress.warn("c", "m")
        assert progress.emit_budget(SimpleNamespace(elapsed_minutes=lambda: 12.5, max_minutes=180))
    assert {t for t, _ in spy.events} <= _V1_TYPES
    assert len(spy.events) == 8
    for event_type, data in spy.events:
        _valid(event_type, data)
    assert spy.events[4][1]["gain_pct"] == 10.0
    assert spy.events[5][1]["gain_pct"] is None


def test_the_plug_host_lock_mismatch_event_matches_the_schema():
    spy = _Spy()
    registry = builtin_registry()
    registry.register(Implementation("events", "spy", lambda ctx: spy))
    manifest = ResolvedManifest(
        "p", {"events": SocketSetting("spy", {}, "test")}, (LockMismatch("measurement", "local", "hosted"),)
    )
    PlugHost(manifest, registry, session_dir=None).start()
    assert [t for t, _ in spy.events] == [EVENT_LOCK_MISMATCH]
    _valid(*spy.events[0])


def test_a_host_without_an_events_socket_drops_the_event():
    host = PlugHost(ResolvedManifest("p", {}, ()), builtin_registry(), session_dir=None)
    host.start()
    with activated(host):
        progress.warn("c", "m")


def test_a_failing_events_implementation_reaches_the_caller():
    with activated(_host(_Raising())), pytest.raises(RuntimeError, match="^sink is broken$"):
        progress.warn("c", "m")


def test_budget_goes_out_at_most_every_five_minutes(monkeypatch):
    spy = _Spy()
    now = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    with activated(_host(spy)):
        assert progress.emit_budget(_state(1))
        now[0] += progress.BUDGET_INTERVAL_SEC - 1
        assert not progress.emit_budget(_state(2))
        now[0] += 1
        assert progress.emit_budget(_state(3))
    assert [d["elapsed_min"] for _, d in spy.events] == [1, 3]
