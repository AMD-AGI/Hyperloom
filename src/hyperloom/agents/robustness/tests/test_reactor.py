# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end reactor tests. The subprocess-transport JSON-IO contract is exercised in test_runtime_cli.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.agents.robustness.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
    Finding,
)
from hyperloom.agents.robustness.decision.policy_aware import PolicyAware
from hyperloom.agents.robustness.role.envelope import IntentType
from hyperloom.agents.robustness.role.findings import FindingSink, FindingSinkConfig
from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.role.reactor import Reactor, ReactorComponents
from hyperloom.agents.robustness.signals import Classifier
from hyperloom.agents.robustness.signals.crash import CrashConfig
from hyperloom.agents.robustness.sources.base import (
    DegradeRouter,
    SourceData,
    SourceUnavailable,
)


class _FakeSource:
    """Stub source that returns a fixed snapshot or raises a fixed exception."""

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
    fb = fallback or _FakeSource("fb", SourceData(coordinator_events=[], sources_used=["fb"]))
    router = DegradeRouter(primary, fb, fail_threshold=2, recheck_interval_s=0.0)
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    components = ReactorComponents(
        router=router,
        classifier=classifier or Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=cooldown_ticks)),
        policy=PolicyAware(),
        sink=sink,
    )
    return Reactor(components), sink


def _ctx(crash_count: int = 0) -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(crash_count=crash_count),
        inbox=[],
        now_unix=1.0,
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
    primary = _FakeSource("local-probe", SourceUnavailable("down"))
    fallback = _FakeSource(
        "quiet-fallback",
        SourceData(
            local_log_errors=[{"pattern": "CUDA out of memory", "line": "boom"}],
            sources_used=["quiet-fallback"],
        ),
    )
    reactor, _ = _build_reactor(primary=primary, fallback=fallback, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx())
    assert primary.calls == 1
    assert fallback.calls == 1
    assert any(i.type is IntentType.ALERT for i in intents)
    assert any(i.payload.get("severity") == "high" for i in intents if i.type is IntentType.ALERT)


# ---------------------------------------------------------------------------
# Policy filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactor_drops_invalid_intents_emitted_by_extra_evaluator(tmp_path: Path):
    from hyperloom.agents.robustness.role.envelope import Intent, IntentType

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
        classifier=Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=CustomLadder(),
        policy=PolicyAware(),
        sink=None,
    )
    reactor = Reactor(components)
    intents = await reactor.tick(_ctx(crash_count=2))
    assert all(i.payload.get("severity") for i in intents if i.type is IntentType.ALERT)


# ---------------------------------------------------------------------------
# FindingSink unit tests (folded in from test_findings_sink.py)
# ---------------------------------------------------------------------------


def _finding(**overrides) -> Finding:
    base = dict(
        tick_index=1,
        timestamp_unix=1.0,
        symptom_name="x",
        severity="medium",
        summary="s",
        intents=[{"intent_type": "alert", "payload": {"severity": "medium", "summary": "s"}}],
        evidence={"k": 1},
        rca_text="",
    )
    base.update(overrides)
    return Finding(**base)


@pytest.mark.asyncio
async def test_sink_appends_jsonl_rows(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    written = await sink.append_many([_finding(tick_index=1), _finding(tick_index=2)])
    assert written == 2
    path = sink.file_path
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert [r["tick_index"] for r in rows] == [1, 2]
    assert all(r["intents"][0]["intent_type"] == "alert" for r in rows)


@pytest.mark.asyncio
async def test_sink_appends_across_calls(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-2"))
    await sink.append_many([_finding(tick_index=1)])
    await sink.append_many([_finding(tick_index=2)])
    rows = sink.file_path.read_text().splitlines()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_sink_creates_subdirectories_when_missing(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="abc"))
    assert not sink.file_path.exists()
    await sink.append_many([_finding()])
    assert sink.file_path.exists()
    assert sink.file_path.parent.name == "findings"


@pytest.mark.asyncio
async def test_sink_is_resilient_to_io_errors(tmp_path: Path, caplog):
    # Block creation by pre-creating a file where the parent dir would go
    blocker = tmp_path / "blocked"
    blocker.write_text("dummy")
    cfg2 = FindingSinkConfig(session_dir=blocker / "x", session_id="sess")
    sink2 = FindingSink(cfg2)
    with caplog.at_level("WARNING"):
        written = await sink2.append_many([_finding()])
    assert written == 1
    assert any("findings sink" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_sink_no_op_on_empty_iterable(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess"))
    assert await sink.append_many([]) == 0
    assert not sink.file_path.exists()
