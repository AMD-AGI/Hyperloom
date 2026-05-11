"""End-to-end reactor + backend adapter tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness_agent.decision.action_ladder import ActionLadder, ActionLadderConfig
from robustness_agent.decision.policy_aware import PolicyAware
from robustness_agent.findings.sink import FindingSink, FindingSinkConfig
from robustness_agent.role.envelope import IntentType
from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.role.reactor import Reactor, ReactorComponents
from robustness_agent.signals import Classifier
from robustness_agent.signals.crash import CrashConfig
from robustness_agent.sources.base import (
    DegradeRouter,
    SourceData,
    SourceUnavailable,
)


class _FakeSource:
    def __init__(self, name: str, snapshot: SourceData | Exception):
        self.name = name
        self._snapshot = snapshot
        self.calls = 0

    async def fetch(self, ctx) -> SourceData:
        self.calls += 1
        if isinstance(self._snapshot, BaseException):
            raise self._snapshot
        return self._snapshot


def _build_reactor(
    *,
    primary: _FakeSource,
    fallback: _FakeSource | None = None,
    tmp_path: Path,
    classifier: Classifier | None = None,
    cooldown_ticks: int = 0,
) -> tuple[Reactor, FindingSink]:
    fb = fallback or _FakeSource(
        "fb", SourceData(coordinator_events=[], sources_used=["fb"])
    )
    router = DegradeRouter(primary, fb, fail_threshold=2, recheck_interval_s=0.0)
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    components = ReactorComponents(
        router=router,
        classifier=classifier or Classifier(crash_config=CrashConfig(medium_threshold=2)),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=cooldown_ticks)),
        policy=PolicyAware(),
        sink=sink,
    )
    return Reactor(components), sink


def _ctx(crash_count: int = 0, *, session_id: str = "sess-1", now_unix: float = 1.0) -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(session_id=session_id, crash_count=crash_count),
        inbox=[],
        now_unix=now_unix,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reactor_emits_heartbeat_when_no_symptoms(tmp_path: Path):
    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    reactor, sink = _build_reactor(primary=primary, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx())
    assert len(intents) == 1
    assert intents[0].type is IntentType.SEND_MESSAGE
    assert intents[0].payload["topic"] == "heartbeat"
    assert reactor.tick_index == 1
    assert not sink.file_path.exists()


@pytest.mark.asyncio
async def test_reactor_emits_alert_for_crash_count_and_persists_finding(tmp_path: Path):
    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    reactor, sink = _build_reactor(primary=primary, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx(crash_count=2))
    assert any(i.type is IntentType.ALERT for i in intents)
    assert sink.file_path.exists()
    rows = sink.file_path.read_text().splitlines()
    assert rows
    row = json.loads(rows[0])
    assert row["symptom_name"] == "crash_count_rising"
    assert row["intents"][0]["intent_type"] == "alert"


# ---------------------------------------------------------------------------
# DegradeRouter integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reactor_falls_back_to_secondary_when_primary_fails(tmp_path: Path):
    primary = _FakeSource("server", SourceUnavailable("down"))
    fallback = _FakeSource(
        "local",
        SourceData(
            session_pods=[{"pod": {"namespace": "ns", "name": "p"}, "phase": "Failed"}],
            sources_used=["local"],
        ),
    )
    reactor, _ = _build_reactor(primary=primary, fallback=fallback, tmp_path=tmp_path)
    # Two consecutive failures degrade primary; we only need one tick to see
    # fallback used because fail_threshold=1 would degrade immediately. With
    # threshold=2 the first tick still falls through to fallback after primary
    # exception.
    intents = await reactor.tick(_ctx())
    assert any(i.type is IntentType.ALERT for i in intents)
    assert any(i.payload.get("severity") == "high" for i in intents if i.type is IntentType.ALERT)


# ---------------------------------------------------------------------------
# Policy filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reactor_drops_invalid_intents_emitted_by_extra_evaluator(tmp_path: Path):
    from robustness_agent.role.envelope import Intent, IntentType
    from robustness_agent.signals import Symptom, SymptomSeverity

    class CustomLadder(ActionLadder):
        def _intents_for(self, sym):  # type: ignore[override]
            base = super()._intents_for(sym)
            base.append(Intent(type=IntentType.ALERT, payload={"summary": "no severity"}))
            return base

    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    fallback = _FakeSource("local", SourceData(sources_used=["local"]))
    router = DegradeRouter(primary, fallback, fail_threshold=2, recheck_interval_s=0.0)
    components = ReactorComponents(
        router=router,
        classifier=Classifier(crash_config=CrashConfig(medium_threshold=2)),
        ladder=CustomLadder(),
        policy=PolicyAware(),
        sink=None,
    )
    reactor = Reactor(components)
    intents = await reactor.tick(_ctx(crash_count=2))
    # The bogus intent should be filtered, keeping only the canonical alert.
    assert all(i.payload.get("severity") for i in intents if i.type is IntentType.ALERT)


# Backend-adapter level tests (subprocess transport: prompt -> ReactorContext
# parsing + reactor tick advancement) live in test_runtime_cli.py — that's
# where the host-visible JSON-IO contract is exercised end-to-end.
